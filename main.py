import asyncio
import datetime
import logging
import os
import sqlite3
import subprocess
import sys
import traceback
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from collections import defaultdict, deque

from market_data.candle_fetcher import (
    Candle,
    start_candle_stream,
    initialize_history,
)
from market_data import spread_monitor, tick_window
from indicators.factor_cache import all_factors, on_candle
from analysis.regime_classifier import classify
from analysis.focus_decider import decide_focus
from analysis.gpt_decider import get_decision, fallback_decision
from analysis.local_decider import heuristic_decision
from analysis.perf_monitor import snapshot as get_perf
from analysis.summary_ingestor import check_event_soon, get_latest_news
from analysis.macro_state import MacroState
from analysis.macro_snapshot_builder import (
    DEFAULT_SNAPSHOT_PATH as MACRO_SNAPSHOT_DEFAULT_PATH,
    refresh_macro_snapshot,
)
try:
    from analysis.kaizen import audit_loop as kaizen_loop  # type: ignore
except ModuleNotFoundError:
    kaizen_loop = None
# バックグラウンドでニュース取得と要約を実行するためのインポート
from market_data.news_fetcher import fetch_loop as news_fetch_loop
from analysis.summary_ingestor import ingest_loop as summary_ingest_loop
from analytics.insight_client import InsightClient
from analysis.range_guard import detect_range_mode
from analysis.pattern_stats import PatternStats, derive_pattern_signature
from signals.pocket_allocator import alloc, DEFAULT_SCALP_SHARE, dynamic_scalp_share
from execution.risk_guard import (
    allowed_lot,
    build_exposure_state,
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
from execution.stage_rules import compute_stage_lot
from execution.order_ids import build_client_order_id
from execution.managed_positions import filter_bot_managed_positions
from execution.pocket_limits import (
    POCKET_ENTRY_MIN_INTERVAL,
    POCKET_EXIT_COOLDOWNS,
    POCKET_LOSS_COOLDOWNS,
    cooldown_for_pocket,
)
from analytics.realtime_metrics_client import (
    ConfidencePolicy,
    RealtimeMetricsClient,
    StrategyHealth,
)
from analysis import policy_bus, plan_bus
from strategies.trend.ma_cross import MovingAverageCross
from strategies.trend.h1_momentum import H1MomentumSwing
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.news.spike_reversal import NewsSpikeReversal
from strategies.scalping.m1_scalper import M1Scalper
from strategies.scalping.range_fader import RangeFader
from strategies.scalping.pulse_break import PulseBreak
from strategies.scalping.impulse_retrace import ImpulseRetraceScalp
from strategies.micro.momentum_burst import MomentumBurstMicro
from strategies.micro.trend_momentum import TrendMomentumMicro
from strategies.micro.momentum_stack import MicroMomentumStack
from strategies.micro.pullback_ema import MicroPullbackEMA
from strategies.micro.range_break import MicroRangeBreak
from strategies.micro.level_reactor import MicroLevelReactor
from utils.oanda_account import get_account_snapshot
from utils.secrets import get_secret
from utils.market_hours import is_market_open
from utils.hold_monitor import HoldMonitor
from utils.metrics_logger import log_metric
import workers.impulse_break_s5.config as impulse_break_s5_config
import workers.impulse_retest_s5.config as impulse_retest_s5_config
import workers.impulse_momentum_s5.config as impulse_momentum_s5_config
import workers.manual_swing.config as manual_swing_config
import workers.mirror_spike.config as mirror_spike_config
import workers.mirror_spike_s5.config as mirror_spike_s5_config
import workers.trend_h1.config as trend_h1_config
import workers.mirror_spike_tight.config as mirror_spike_tight_config
import workers.pullback_s5.config as pullback_s5_config
import workers.macro_core.config as macro_core_config
try:
    import workers.pullback_runner_s5.config as pullback_runner_s5_config
    from workers.pullback_runner_s5 import pullback_runner_s5_worker
except ModuleNotFoundError:
    import asyncio as _pr_asyncio
    class _PRDummy:
        ENABLED = False
        LOG_PREFIX = "[PULLBACK-RUNNER-S5]"
    pullback_runner_s5_config = _PRDummy()
    async def pullback_runner_s5_worker():
        await _pr_asyncio.sleep(0.0)
import workers.pullback_scalp.config as pullback_scalp_config
import workers.scalp_exit.config as scalp_exit_config
import workers.scalp_core.config as scalp_core_config
import workers.squeeze_break_s5.config as squeeze_break_s5_config
import workers.vwap_magnet_s5.config as vwap_magnet_s5_config
import workers.micro_core.config as micro_core_config
import workers.london_momentum.config as london_momentum_config
from workers.impulse_break_s5 import impulse_break_s5_worker
from workers.impulse_momentum_s5 import impulse_momentum_s5_worker
from workers.impulse_retest_s5 import impulse_retest_s5_worker
from workers.london_momentum import london_momentum_worker
from workers.manual_swing import manual_swing_worker
from workers.mirror_spike import mirror_spike_worker
from workers.mirror_spike_s5 import mirror_spike_s5_worker
from workers.mirror_spike_tight import mirror_spike_tight_worker
from workers.pullback_s5 import pullback_s5_worker
from workers.pullback_scalp import pullback_scalp_worker
from workers.scalp_exit import scalp_exit_worker
from workers.squeeze_break_s5 import squeeze_break_s5_worker
from workers.trend_h1 import trend_h1_worker
from workers.vwap_magnet_s5 import vwap_magnet_s5_worker
import workers.onepip_maker_s1.config as onepip_maker_s1_config
from workers.onepip_maker_s1 import onepip_maker_s1_worker
from workers.micro_core import micro_core_worker
from workers.macro_core import macro_core_worker
from workers.scalp_core import scalp_core_worker
from workers.common.pocket_plan import PocketPlan

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Optional file logging (set FILE_LOG_PATH to enable)
try:
    _file_log = os.getenv("FILE_LOG_PATH")
    if _file_log:
        _fh = logging.FileHandler(_file_log)
        _fh.setLevel(logging.INFO)
        _fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(_fh)
except Exception:
    pass

logging.info("Application started!")

STRATEGIES = {
    "TrendMA": MovingAverageCross,
    "Donchian55": Donchian55,
    "H1Momentum": H1MomentumSwing,
    "BB_RSI": BBRsi,
    "NewsSpikeReversal": NewsSpikeReversal,
    "M1Scalper": M1Scalper,
    "RangeFader": RangeFader,
    "PulseBreak": PulseBreak,
    "ImpulseRetrace": ImpulseRetraceScalp,
    "MomentumBurst": MomentumBurstMicro,
    "TrendMomentumMicro": TrendMomentumMicro,
    "MicroMomentumStack": MicroMomentumStack,
    "MicroPullbackEMA": MicroPullbackEMA,
    "MicroRangeBreak": MicroRangeBreak,
    "MicroLevelReactor": MicroLevelReactor,
}

POCKET_STRATEGY_MAP = {
    "macro": {"TrendMA", "Donchian55"},
    "micro": {
        "BB_RSI",
        "NewsSpikeReversal",
        "MomentumBurst",
        "TrendMomentumMicro",
        "MicroMomentumStack",
        "MicroPullbackEMA",
        "MicroRangeBreak",
        "MicroLevelReactor",
    },
    "scalp": {"M1Scalper", "RangeFader", "PulseBreak", "ImpulseRetrace"},
}

MANAGED_POCKETS = {"macro", "micro", "scalp", "scalp_fast"}

FOCUS_POCKETS = {
    "macro": ("macro",),
    "micro": ("micro", "scalp"),
    "hybrid": ("macro", "micro", "scalp"),
    "event": ("macro", "micro"),
}


def _micro_flow_factor(
    spread_pips: Optional[float],
    margin_buffer: Optional[float],
) -> float:
    factor = 1.0
    try:
        spread_val = float(spread_pips) if spread_pips is not None else None
    except (TypeError, ValueError):
        spread_val = None
    if spread_val is not None:
        warn, alert, block = MICRO_SPREAD_DAMP_PIPS
        warn_factor, alert_factor, block_factor = MICRO_SPREAD_DAMP_FACTORS
        if spread_val >= block:
            factor *= block_factor
        elif spread_val >= alert:
            factor *= alert_factor
        elif spread_val >= warn:
            factor *= warn_factor
    if margin_buffer is not None and margin_buffer < MICRO_MARGIN_BUFFER_LIMIT:
        factor *= MICRO_MARGIN_BUFFER_FACTOR
    return max(0.0, min(1.0, factor))


async def ensure_factor_history_ready(state: Optional[dict]) -> None:
    """Placeholder warmup hook; returns immediately if no warmup is configured."""
    if not state:
        return None
    return None

# Manual-trade guard: by default, do not let the bot exit or partially close
# trades that were entered manually (i.e. trades without a QuantRabbit
# clientOrderID like "qr-..."). This only affects exit/partial logic and
# dynamic protection updates; exposure accounting still includes all trades.
TRENDMA_FAST_LOSS_THRESHOLD_SEC = 300
TRENDMA_FAST_LOSS_BASE_COOLDOWN = 600
TRENDMA_FAST_LOSS_STREAK_STEP = 60
TRENDMA_NEWS_FREEZE_SECONDS = 420
TRENDMA_FLIP_LOOKBACK_SEC = 600
TRENDMA_FLIP_STRENGTH_RATIO = 0.5
TRENDMA_FLIP_SLOPE_MIN = 0.05
TRENDMA_FLIP_ADX_MIN = 18.0

if os.getenv("SCALP_TACTICAL", "0").strip().lower() not in {"", "0", "false", "no"}:
    POCKET_LOSS_COOLDOWNS["scalp"] = 120
    POCKET_ENTRY_MIN_INTERVAL["scalp"] = 25

PIP = 0.01

POCKET_MAX_ACTIVE_TRADES = {
    "macro": 6,
    "micro": 6,
    "scalp": 4,
    "scalp_fast": 3,
}

POCKET_MAX_ACTIVE_TRADES_RANGE = {
    "macro": 0,
    "micro": 2,
    "scalp": 2,
    "scalp_fast": 1,
}

POCKET_MAX_DIRECTIONAL_TRADES = {
    "macro": 6,
    "micro": 3,
    "scalp": 2,
    "scalp_fast": 2,
}

POCKET_MAX_DIRECTIONAL_TRADES_RANGE = {
    "macro": 0,
    "micro": 1,
    "scalp": 1,
    "scalp_fast": 1,
}

RANGE_MACRO_BIAS_MAX_ACTIVE_TRADES = 2
RANGE_MACRO_BIAS_MAX_DIRECTIONAL_TRADES = 2

# News cache fetch limits (short/long list)
NEWS_LIMITS = {"short": 2, "long": 2}

try:
    HEDGING_ENABLED = get_secret("oanda_hedging_enabled").lower() == "true"
except Exception:
    HEDGING_ENABLED = False
if HEDGING_ENABLED:
    logging.info("[CONFIG] Hedging enabled; allowing offsetting positions.")

FALLBACK_EQUITY = 10000.0  # REST失敗時のフォールバック

RANGE_MACRO_WEIGHT_CAP = 0.22
RANGE_CONFIDENCE_SCALE = {
    "macro": 0.65,
    "micro": 0.85,
    "scalp": 0.75,
}
RANGE_TREND_CONFIDENCE_DAMP = {
    "TrendMA": 0.65,
    "MomentumBurst": 0.75,
    "TrendMomentumMicro": 0.75,
    "MicroMomentumStack": 0.8,
}
RANGE_SCALP_ATR_MIN = 1.8
RANGE_SCALP_MOMENTUM_MIN = 0.001
RANGE_SCALP_VOL_MIN = 0.9
SOFT_RANGE_SUPPRESS_STRATEGIES = {"TrendMA", "Donchian55"}
LOW_TREND_ADX_THRESHOLD = 18.0
LOW_TREND_SLOPE_THRESHOLD = 0.00035
LOW_TREND_WEIGHT_CAP = 0.35
MACRO_LOT_BOOST = float(os.getenv("MACRO_LOT_BOOST", "1.25"))
MACRO_LOT_BOOST_WEIGHT = float(os.getenv("MACRO_LOT_BOOST_WEIGHT", "0.15"))
MACRO_LOT_BIAS_MIN = float(os.getenv("MACRO_LOT_BIAS_MIN", "0.2"))
MACRO_LOT_REDUCTION_FACTOR = float(os.getenv("MACRO_LOT_REDUCTION_FACTOR", "0.75"))
MACRO_LOT_MIN_FRACTION = float(os.getenv("MACRO_LOT_MIN_FRACTION", "0.12"))
MACRO_LOT_DISABLE_THRESHOLD = float(os.getenv("MACRO_LOT_DISABLE_THRESHOLD", "0.05"))
MICRO_BIAS_REDUCTION_THRESHOLD = float(os.getenv("MICRO_BIAS_REDUCTION_THRESHOLD", "0.2"))
MICRO_BIAS_DISABLE_THRESHOLD = float(os.getenv("MICRO_BIAS_DISABLE_THRESHOLD", "0.08"))
MICRO_BIAS_REDUCTION_FACTOR = float(os.getenv("MICRO_BIAS_REDUCTION_FACTOR", "0.4"))
TARGET_MARGIN_USAGE = float(os.getenv("TARGET_MARGIN_USAGE", "0.92"))
MAX_MARGIN_USAGE_BOOST = float(os.getenv("MAX_MARGIN_USAGE_BOOST", "1.35"))
LOSS_GUARD_EXPAND_ATR_MIN = float(os.getenv("LOSS_GUARD_EXPAND_ATR_MIN", "1.2"))
LOSS_GUARD_EXPAND_RATIO = float(os.getenv("LOSS_GUARD_EXPAND_RATIO", "1.2"))
LOSS_GUARD_MAX_PIPS = float(os.getenv("LOSS_GUARD_MAX_PIPS", "6.0"))
MIN_HOLD_SEC_PER_ATR = float(os.getenv("MIN_HOLD_SEC_PER_ATR", "30.0"))
PRICE_SURGE_WINDOW_SEC = int(os.getenv("PRICE_SURGE_WINDOW_SEC", "600"))
PRICE_SURGE_MIN_MOVE = float(os.getenv("PRICE_SURGE_MIN_MOVE", "0.30"))
PRICE_SURGE_ATR_MIN = float(os.getenv("PRICE_SURGE_ATR_MIN", "1.5"))
PRICE_SURGE_VOL_MIN = float(os.getenv("PRICE_SURGE_VOL_MIN", "1.0"))
PRICE_SURGE_COOLDOWN_SEC = int(os.getenv("PRICE_SURGE_COOLDOWN_SEC", "180"))
PRICE_SURGE_BLOCK_SEC = int(os.getenv("PRICE_SURGE_BLOCK_SEC", "900"))
# 長めの監視窓（Acceptance: 30分±20pips）も検知対象にする
PRICE_SURGE_LONG_WINDOW_SEC = int(os.getenv("PRICE_SURGE_LONG_WINDOW_SEC", "1800"))
PRICE_SURGE_LONG_MIN_MOVE = float(os.getenv("PRICE_SURGE_LONG_MIN_MOVE", "0.20"))
SPREAD_OVERRIDE_MACRO_PIPS = float(os.getenv("SPREAD_OVERRIDE_MACRO_PIPS", "1.40"))
SPREAD_OVERRIDE_MICRO_PIPS = float(os.getenv("SPREAD_OVERRIDE_MICRO_PIPS", "1.15"))
MACRO_STALE_DISABLE_SEC = float(os.getenv("MACRO_STALE_DISABLE_SEC", "2400"))
MICRO_SPREAD_DAMP_PIPS = (
    float(os.getenv("MICRO_SPREAD_WARN_PIPS", "0.80")),
    float(os.getenv("MICRO_SPREAD_ALERT_PIPS", "1.00")),
    float(os.getenv("MICRO_SPREAD_BLOCK_PIPS", "1.20")),
)
MICRO_SPREAD_DAMP_FACTORS = (
    float(os.getenv("MICRO_SPREAD_WARN_FACTOR", "0.75")),
    float(os.getenv("MICRO_SPREAD_ALERT_FACTOR", "0.50")),
    float(os.getenv("MICRO_SPREAD_BLOCK_FACTOR", "0.30")),
)
MICRO_MARGIN_BUFFER_LIMIT = float(os.getenv("MICRO_MARGIN_BUFFER_LIMIT", "0.04"))
MICRO_MARGIN_BUFFER_FACTOR = float(os.getenv("MICRO_MARGIN_BUFFER_FACTOR", "0.50"))
MICRO_MARGIN_GUARD_BUFFER = float(os.getenv("MICRO_MARGIN_GUARD_BUFFER", "0.025"))
MICRO_MARGIN_GUARD_RELEASE = float(
    os.getenv("MICRO_MARGIN_GUARD_RELEASE", "0.035")
)
MICRO_MARGIN_GUARD_STOP = float(os.getenv("MICRO_MARGIN_GUARD_STOP", "0.018"))
SOFT_RANGE_SCORE_MIN = 0.58
SOFT_RANGE_COMPRESSION_MIN = 0.55
SOFT_RANGE_VOL_MIN = 0.40

DEFAULT_MIN_HOLD_SEC = {
    "macro": 360.0,  # 6 min baseline for swing legs
    "micro": 150.0,  # 2.5 min baseline for pullback setups
    "scalp": 75.0,   # legacy scalps still need >1 min to settle
}
DEFAULT_LOSS_GUARD_PIPS = {
    "macro": 3.6,
    "micro": 1.4,
    "scalp": 2.0,
}
MANUAL_SENTINEL_POCKETS = set()
MANUAL_SENTINEL_MIN_UNITS = int(os.getenv("MANUAL_SENTINEL_MIN_UNITS", "0"))

def _env_set(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name)
    if raw is None:
        raw = default
    return {item.strip() for item in raw.split(",") if item.strip()}

MANUAL_SENTINEL_BLOCK_POCKETS = _env_set("MANUAL_SENTINEL_BLOCK_POCKETS", "")
MANUAL_SENTINEL_RELEASE_CYCLES = max(
    1, int(os.getenv("MANUAL_SENTINEL_RELEASE_CYCLES", "1"))
)
MANUAL_SENTINEL_STALE_RELEASE_SEC = max(
    0.0, float(os.getenv("MANUAL_SENTINEL_STALE_RELEASE_SEC", "90"))
)
_MANUAL_SENTINEL_ACTIVE = False
_MANUAL_SENTINEL_CLEAR_STREAK = 0
HOLD_RATIO_LOOKBACK_HOURS = float(os.getenv("HOLD_RATIO_LOOKBACK_HOURS", "6.0"))
HOLD_RATIO_MIN_SAMPLES = int(os.getenv("HOLD_RATIO_MIN_SAMPLES", "80"))
HOLD_RATIO_MAX = float(os.getenv("HOLD_RATIO_MAX", "0.30"))
HOLD_RATIO_RELEASE_FACTOR = float(os.getenv("HOLD_RATIO_RELEASE_FACTOR", "0.8"))
HOLD_RATIO_CHECK_INTERVAL_SEC = float(os.getenv("HOLD_RATIO_CHECK_INTERVAL_SEC", "900"))
HOLD_RATIO_BLOCK_POCKETS = {"micro", "scalp"}
_HOLD_RATIO_GUARD_ACTIVE = False
_HOLD_MONITOR = HoldMonitor(
    db_path=Path(os.getenv("HOLD_RATIO_DB", "logs/trades.db")),
    lookback_hours=HOLD_RATIO_LOOKBACK_HOURS,
    min_samples=HOLD_RATIO_MIN_SAMPLES,
)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _atr_hint_pips(fac_m1: Dict[str, Any]) -> float:
    atr = fac_m1.get("atr_pips")
    if atr is None:
        raw = fac_m1.get("atr")
        if raw is not None:
            try:
                atr = float(raw) * 100.0
            except (TypeError, ValueError):
                atr = None
    return float(atr or 6.0)


def _derive_min_hold_seconds(
    signal: Dict[str, Any],
    pocket: str,
    fac_m1: Dict[str, Any],
) -> float:
    explicit = signal.get("min_hold_sec") or signal.get("min_hold_seconds")
    val = _to_float(explicit)
    if val is not None:
        return max(0.0, val)
    base = DEFAULT_MIN_HOLD_SEC.get(pocket, 120.0 if pocket == "micro" else 90.0)
    tp = _to_float(signal.get("tp_pips")) or 0.0
    atr_hint = _atr_hint_pips(fac_m1)
    # Scale baseline by TP demand and prevailing ATR so larger swings get more time.
    scaled = base
    if tp > 0:
        scaled = max(scaled, min(600.0, tp * 45.0))
    if atr_hint > 0 and MIN_HOLD_SEC_PER_ATR > 0:
        scaled = max(scaled, min(600.0, atr_hint * MIN_HOLD_SEC_PER_ATR))
    return round(scaled, 1)


def _derive_loss_guard_pips(signal: Dict[str, Any], pocket: str, fac_m1: Dict[str, Any]) -> float:
    explicit = signal.get("loss_guard_pips") or signal.get("loss_grace_pips")
    val = _to_float(explicit)
    if val is not None:
        return max(0.1, val)
    baseline = DEFAULT_LOSS_GUARD_PIPS.get(pocket, 1.0)
    tp = _to_float(signal.get("tp_pips")) or 0.0
    sl = _to_float(signal.get("sl_pips")) or 0.0
    guard = baseline
    if tp > 0:
        guard = max(guard, tp * 0.35)
    if sl > 0:
        guard = min(guard, sl * 0.7)
    guard = max(0.3, guard)
    if pocket in {"macro", "micro"}:
        atr_hint = _atr_hint_pips(fac_m1)
        vol_raw = fac_m1.get("vol_5m")
        try:
            vol_recent = float(vol_raw) if vol_raw is not None else None
        except (TypeError, ValueError):
            vol_recent = None
    if pocket in {"macro", "micro"}:
        if atr_hint and atr_hint >= LOSS_GUARD_EXPAND_ATR_MIN:
            expanded = min(LOSS_GUARD_MAX_PIPS, atr_hint * LOSS_GUARD_EXPAND_RATIO)
            guard = max(guard, expanded)
    return round(guard, 2)


def _parse_trade_entry_thesis(trade: Dict[str, Any]) -> Dict[str, Any]:
    thesis = trade.get("entry_thesis")
    if isinstance(thesis, str):
        try:
            thesis = json.loads(thesis)
        except Exception:
            thesis = {}
    if not isinstance(thesis, dict):
        thesis = {}
    return thesis


def _strategy_position_snapshot(
    open_info: Dict[str, Any] | None,
    direction: str,
    strategy_tag: Optional[str],
    fallback_price: Optional[float],
) -> tuple[int, Optional[float]]:
    if not open_info:
        return 0, fallback_price
    key_units = "long_units" if direction == "long" else "short_units"
    key_avg = "long_avg_price" if direction == "long" else "short_avg_price"
    if not strategy_tag:
        try:
            units = int(open_info.get(key_units, 0) or 0)
        except (TypeError, ValueError):
            units = 0
        avg = open_info.get(key_avg)
        if avg is None:
            avg = fallback_price
        return max(units, 0), avg
    trades = open_info.get("open_trades") or []
    total_units = 0
    weighted_price = 0.0
    for tr in trades:
        if tr.get("side") != direction:
            continue
        tag = tr.get("strategy_tag")
        if not tag:
            thesis = _parse_trade_entry_thesis(tr)
            tag = thesis.get("strategy_tag") or thesis.get("strategy")
        if tag != strategy_tag:
            continue
        try:
            units = abs(int(tr.get("units", 0) or 0))
        except (TypeError, ValueError):
            units = 0
        if units <= 0:
            continue
        total_units += units
        price_raw = tr.get("price")
        try:
            price_val = float(price_raw)
        except (TypeError, ValueError):
            price_val = fallback_price
        if price_val is not None:
            weighted_price += price_val * units
    avg_price = fallback_price
    if total_units and weighted_price > 0.0:
        avg_price = weighted_price / total_units
    elif total_units and avg_price is None:
        avg_price = fallback_price
    return total_units, avg_price


def _extract_profile_name(raw_signal: Dict[str, Any], strategy_name: str) -> str:
    profile = raw_signal.get("profile")
    if isinstance(profile, str) and profile.strip():
        return profile.strip()
    return strategy_name


def _manual_sentinel_state(open_positions: Dict[str, Dict]) -> tuple[bool, int, str]:
    # Manual sentinel is fully disabled per user request.
    return False, 0, ""
SOFT_RANGE_WEIGHT_CAP = 0.32
SOFT_RANGE_ADX_BUFFER = 6.0
RANGE_ENTRY_CONFIRMATIONS = 1
RANGE_EXIT_CONFIRMATIONS = 3
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
TARGET_INSTRUMENT = "USD_JPY"
# 再起動直後の暴発を避ける猶予時間（秒）
COLD_START_GRACE_SEC = int(os.getenv("COLD_START_GRACE_SEC", "90"))
IDLE_REFRESH_THRESHOLD_SEC = int(os.getenv("IDLE_REFRESH_THRESHOLD_SEC", "1800"))
IDLE_REFRESH_CHECK_SEC = int(os.getenv("IDLE_REFRESH_CHECK_SEC", "180"))

# 総合的な Macro 配分上限（lot 按分に直接適用）
# 初期値は 0.38 (=38%)。環境変数による上書きは下段で適用。
GLOBAL_MACRO_WEIGHT_CAP = 0.38


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default

# 環境変数 MACRO_WEIGHT_CAP で上書き可能。
GLOBAL_MACRO_WEIGHT_CAP = _env_float("MACRO_WEIGHT_CAP", GLOBAL_MACRO_WEIGHT_CAP)

# Disable sending broker-level stop loss (risk sizing may still use SL pips)
# Default true so all new orders go out without broker SL unless explicitly re-enabled.
DISABLE_STOP_LOSS = _env_bool("DISABLE_STOP_LOSS", True)


_MACRO_STATE_PATH_RAW = (
    os.getenv("MACRO_STATE_SNAPSHOT_PATH")
    or os.getenv("MACRO_STATE_PATH")
    or os.getenv("MACRO_STATE_SNAPSHOT")
)
_MACRO_STATE_PATH = _MACRO_STATE_PATH_RAW or str(MACRO_SNAPSHOT_DEFAULT_PATH)
_MACRO_STATE_DEADZONE = _env_float("MACRO_STATE_DEADZONE", 0.25)
_MACRO_STATE_GATE_ENABLED = _env_bool("MACRO_STATE_GATE_ENABLED", False)
_EVENT_WINDOW_ENABLED = _env_bool("EVENT_WINDOW_ENABLED", True)
_EVENT_WINDOW_BEFORE = _env_float("EVENT_WINDOW_BEFORE_HOURS", 2.0)
_EVENT_WINDOW_AFTER = _env_float("EVENT_WINDOW_AFTER_HOURS", 1.0)
_EVENT_WINDOW_RISK_MULTIPLIER = _env_float("EVENT_WINDOW_RISK_MULTIPLIER", 0.3)
_EXPOSURE_USD_LONG_MAX_LOT = _env_float("EXPOSURE_USD_LONG_MAX_LOT", 2.5)
_MACRO_STATE_STALE_WARN_SEC = _env_float("MACRO_STATE_STALE_WARN_SEC", 0.0)
_MACRO_STALE_WEIGHT_CAP = _env_float("MACRO_STALE_WEIGHT_CAP", 0.18)
_MACRO_STALE_WEIGHT_DECAY = _env_float("MACRO_STALE_WEIGHT_DECAY", 0.5)
_MACRO_SNAPSHOT_REFRESH_MINUTES = _env_float("MACRO_SNAPSHOT_REFRESH_MINUTES", 10.0)
_MACRO_AUTO_REFRESH_ON_STALE = _env_bool("MACRO_AUTO_REFRESH_ON_STALE", True)

_macro_state_cache: MacroState | None = None
_macro_state_mtime: float | None = None
_macro_state_stale_warned = False
_macro_state_missing_warned = False
_TUNER_LAST_RUN_TS = 0.0
_DELEGATE_MICRO = _env_bool("MICRO_DELEGATE_TO_WORKER", True)
_DELEGATE_MACRO = _env_bool("MACRO_DELEGATE_TO_WORKER", True)
_DELEGATE_SCALP = _env_bool("SCALP_DELEGATE_TO_WORKER", True)


def _pocket_delegated(pocket: str) -> bool:
    if pocket == "micro":
        return _DELEGATE_MICRO
    if pocket == "macro":
        return _DELEGATE_MACRO
    if pocket == "scalp":
        return _DELEGATE_SCALP
    return False


def _pocket_worker_owns_orders(pocket: str) -> bool:
    if pocket == "macro":
        return _DELEGATE_MACRO
    if pocket == "scalp":
        return _DELEGATE_SCALP
    return False


def _macro_weight_cap(perf_cache: dict) -> float:
    stats = perf_cache.get("macro") or {}
    win_rate = stats.get("win_rate")
    if win_rate is None:
        return GLOBAL_MACRO_WEIGHT_CAP
    cap = 0.18 + 0.04 * (win_rate - 0.5)
    cap = max(0.18, min(float(GLOBAL_MACRO_WEIGHT_CAP), cap))
    return cap
_POLICY_VERSION = 0


def _macro_snapshot_path() -> Path:
    return Path(_MACRO_STATE_PATH).expanduser().resolve()


def _parse_iso8601(ts: str) -> datetime.datetime | None:
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _refresh_macro_state() -> MacroState | None:
    global _macro_state_cache, _macro_state_mtime
    path = _macro_snapshot_path()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        globals_dict = globals()
        if _macro_state_cache is None:
            logging.warning("[MACRO] snapshot path missing: %s; using neutral fallback.", path)
            _macro_state_cache = MacroState.neutral(deadzone=_MACRO_STATE_DEADZONE)
        if not globals_dict.get("_macro_state_missing_warned", False):
            logging.warning("[MACRO] snapshot still missing; continuing with fallback.")
            globals_dict["_macro_state_missing_warned"] = True
        return _macro_state_cache
    if _macro_state_cache is None or mtime != _macro_state_mtime:
        try:
            _macro_state_cache = MacroState.load_json(
                path, deadzone=_MACRO_STATE_DEADZONE
            )
            _macro_state_mtime = mtime
            logging.info(
                "[MACRO] snapshot refreshed (asof=%s deadzone=%.2f)",
                _macro_state_cache.snapshot.asof,
                _MACRO_STATE_DEADZONE,
            )
            globals()["_macro_state_stale_warned"] = False
            globals()["_macro_state_missing_warned"] = False
        except Exception as exc:  # noqa: BLE001
            logging.warning("[MACRO] failed to load snapshot: %s", exc)
            return _macro_state_cache
    return _macro_state_cache

# Hard caps to keep risk bounded even with misconfigured secrets
_HARD_BASE_RISK_CAP = 0.03
_HARD_MAX_RISK_CAP = 0.30

try:
    _BASE_RISK_PCT = float(get_secret("risk_pct"))
    if _BASE_RISK_PCT <= 0:
        _BASE_RISK_PCT = 0.025
except Exception:
    _BASE_RISK_PCT = 0.025
_BASE_RISK_PCT = max(_BASE_RISK_PCT, 0.025)
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

def _dynamic_risk_pct(
    signals: list[dict],
    range_mode: bool,
    weight_macro: float | None,
    macro_state: MacroState | None = None,
    *,
    now: datetime.datetime | None = None,
) -> float:
    if range_mode or not signals or _MAX_RISK_PCT <= _BASE_RISK_PCT:
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
    pct = _BASE_RISK_PCT + (_MAX_RISK_PCT - _BASE_RISK_PCT) * score
    if (
        pct > 0.0
        and macro_state
        and _EVENT_WINDOW_ENABLED
        and macro_state.in_event_window(
            TARGET_INSTRUMENT,
            before_hours=_EVENT_WINDOW_BEFORE,
            after_hours=_EVENT_WINDOW_AFTER,
            now=now or datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc),
        )
    ):
        pct *= max(0.0, min(1.0, _EVENT_WINDOW_RISK_MULTIPLIER))
    return pct


def _maybe_run_online_tuner(now: datetime.datetime) -> None:
    """Trigger the online tuner helper on a slow cadence."""
    global _TUNER_LAST_RUN_TS

    if not _env_bool("TUNER_ENABLE", False):
        return

    interval_sec = max(60, _env_int("TUNER_INTERVAL_SEC", 600))
    elapsed = now.timestamp() - _TUNER_LAST_RUN_TS
    if _TUNER_LAST_RUN_TS and elapsed < interval_sec:
        return

    logs_glob = os.getenv("TUNER_LOGS_GLOB", "tmp/exit_eval*.csv")
    presets_path = os.getenv("TUNER_PRESETS", "config/tuning_presets.yaml")
    overrides_path = os.getenv("TUNER_OVERRIDES", "config/tuning_overrides.yaml")
    window_min = max(1, _env_int("TUNER_WINDOW_MINUTES", 15))
    shadow_mode = _env_bool("TUNER_SHADOW_MODE", False)

    cmd = [
        sys.executable or "python3",
        "scripts/run_online_tuner.py",
        "--logs-glob",
        logs_glob,
        "--presets",
        presets_path,
        "--overrides-out",
        overrides_path,
        "--minutes",
        str(window_min),
    ]
    if shadow_mode:
        cmd.append("--shadow")

    try:
        logging.info(
            "[TUNER] run (logs=%s presets=%s shadow=%s minutes=%d)",
            logs_glob,
            presets_path,
            shadow_mode,
            window_min,
        )
        result = subprocess.run(cmd, check=False, timeout=20)
        if result.returncode != 0:
            logging.warning("[TUNER] command exit code %s", result.returncode)
        _TUNER_LAST_RUN_TS = now.timestamp()
    except Exception as exc:  # noqa: BLE001
        logging.warning("[TUNER] skipped: %s", exc)


def _ma_bias(factors: Optional[dict], *, pip_threshold: float = 0.002) -> str:
    if not factors:
        return "neutral"
    try:
        ma10 = float(factors.get("ma10"))
        ma20 = float(factors.get("ma20"))
    except (TypeError, ValueError):
        return "neutral"
    diff = ma10 - ma20
    if diff > pip_threshold:
        return "long"
    if diff < -pip_threshold:
        return "short"
    return "neutral"


def _confidence_from_perf(perf: Optional[dict]) -> float:
    if not perf:
        return 0.5
    try:
        pf = float(perf.get("pf"))
    except (TypeError, ValueError):
        pf = None
    if pf is None:
        return 0.5
    # Map pf range [0.6, 2.0] -> [0, 1]
    normalised = (pf - 0.6) / 1.4
    return max(0.0, min(1.0, normalised))


def _be_profile_for(pocket: str, *, range_mode: bool) -> dict:
    if pocket == "macro":
        trigger = 6.8 if not range_mode else 5.0
        return {
            "enabled": True,
            "trigger_pips": trigger,
            "cooldown_sec": 90.0,
            "lock_ratio": 0.55,
            "min_lock_pips": 2.6 if not range_mode else 2.0,
        }
    if pocket == "micro":
        trigger = 2.4 if not range_mode else 2.0
        return {
            "enabled": True,
            "trigger_pips": trigger,
            "cooldown_sec": 45.0,
            "lock_ratio": 0.5,
            "min_lock_pips": 0.35,
        }
    # scalp
    trigger = 1.3 if not range_mode else 1.1
    return {
        "enabled": True,
        "trigger_pips": trigger,
        "cooldown_sec": 20.0,
        "lock_ratio": 0.45,
        "min_lock_pips": 0.2,
    }


def _partial_profile_for(pocket: str, *, range_mode: bool) -> dict:
    if pocket == "macro":
        thresholds = [4.2, 6.8] if not range_mode else [3.6, 5.2]
        fractions = [0.4, 0.3]
        return {"thresholds_pips": thresholds, "fractions": fractions, "min_units": 80}
    if pocket == "micro":
        thresholds = [1.6, 3.0] if not range_mode else [1.3, 2.4]
        fractions = [0.45, 0.3]
        return {"thresholds_pips": thresholds, "fractions": fractions, "min_units": 40}
    thresholds = [1.2, 2.2] if not range_mode else [1.0, 1.8]
    fractions = [0.5, 0.3]
    return {"thresholds_pips": thresholds, "fractions": fractions, "min_units": 20}


def _publish_policy_snapshot(
    *,
    focus_tag: str,
    focus_pockets: set[str],
    weight_macro: float,
    macro_regime: str,
    micro_regime: str,
    range_ctx,
    range_active: bool,
    event_soon: bool,
    spread_gate_active: bool,
    spread_gate_reason: str,
    spread_macro_relaxed: bool = False,
    spread_micro_relaxed: bool = False,
    lots: dict[str, float],
    perf_cache: dict,
    managed_positions: dict,
    scalp_share: float,
    risk_pct: float,
    fac_m1: dict,
    fac_h4: dict,
    strategies_by_pocket: Dict[str, list[str]] | None = None,
    micro_hint: list[str] | None = None,
) -> None:
    global _POLICY_VERSION
    _POLICY_VERSION += 1

    strategies_by_pocket = strategies_by_pocket or {}

    existing_snapshot = policy_bus.latest()
    existing_data: dict[str, Any] = {}
    existing_pockets: dict[str, Any] = {}
    if existing_snapshot:
        existing_data = existing_snapshot.to_dict()
        existing_pockets = existing_data.get("pockets") or {}

    notes = {
        "focus_tag": focus_tag,
        "macro_regime": macro_regime,
        "micro_regime": micro_regime,
        "range_reason": getattr(range_ctx, "reason", ""),
        "range_score": getattr(range_ctx, "score", 0.0),
        "spread_reason": spread_gate_reason,
        "scalp_share": scalp_share,
        "risk_pct": risk_pct,
        "spread_macro_relaxed": spread_macro_relaxed,
        "spread_micro_relaxed": spread_micro_relaxed,
    }
    if micro_hint:
        notes["micro_hint"] = list(micro_hint)
    pockets: dict[str, dict] = {}
    for pocket in ("macro", "micro", "scalp"):
        bias = "neutral"
        if pocket == "macro":
            bias = _ma_bias(fac_h4)
        else:
            bias = _ma_bias(fac_m1)

        perf = perf_cache.get(pocket, {})
        current_units = 0
        pos_info = managed_positions.get(pocket) or {}
        try:
            current_units = int(pos_info.get("units") or 0)
        except (TypeError, ValueError):
            current_units = 0
        units_cap = int(round(max(0.0, lots.get(pocket, 0.0)) * 100000))
        entry_allow = (
            pocket in focus_pockets
            and (pocket != "macro" or not range_active)
        )
        spread_override = False
        if spread_gate_active:
            if pocket == "macro" and spread_macro_relaxed:
                spread_override = True
            elif pocket in {"micro", "scalp"} and spread_micro_relaxed:
                spread_override = True
            if not spread_override:
                entry_allow = False
        if pocket in {"micro", "scalp"} and event_soon:
            entry_allow = False
        entry_gates = {
            "allow_new": entry_allow,
            "require_retest": range_active and pocket != "macro",
            "spread_ok": (not spread_gate_active) or spread_override,
            "event_ok": not event_soon or pocket == "macro",
        }
        exit_profile = {
            "reverse_threshold": 80 if pocket == "macro" else 70 if pocket == "micro" else 65,
            "allow_negative_exit": False,
        }

        if pocket == "micro" and _DELEGATE_MICRO:
            entry = existing_pockets.get("micro") or {}
            entry = dict(entry)
            if "bias" not in entry:
                entry["bias"] = bias
            entry.setdefault("confidence", _confidence_from_perf(perf))
            entry.setdefault("strategies", [])
            if strategies_by_pocket.get(pocket):
                entry["strategies"] = list(strategies_by_pocket[pocket])
            if micro_hint and not entry.get("strategies"):
                entry["strategies"] = list(micro_hint)
            pockets[pocket] = entry
            continue

        pockets[pocket] = {
            "enabled": pocket in focus_pockets,
            "bias": bias,
            "confidence": _confidence_from_perf(perf),
            "units_cap": units_cap or None,
            "current_units": current_units,
            "entry_gates": entry_gates,
            "exit_profile": exit_profile,
            "be_profile": _be_profile_for(pocket, range_mode=range_active),
            "partial_profile": _partial_profile_for(pocket, range_mode=range_active),
            "strategies": list(strategies_by_pocket.get(pocket, [])),
        }

    snapshot = policy_bus.PolicySnapshot(
        version=_POLICY_VERSION,
        generated_ts=time.time(),
        air_score=getattr(range_ctx, "score", 0.0),
        uncertainty=1.0 if event_soon else 0.0,
        event_lock=event_soon,
        range_mode=range_active,
        notes=notes,
        pockets=pockets,
    )
    policy_bus.publish(snapshot)


def _build_pocket_plan(
    *,
    now: datetime.datetime,
    pocket: str,
    focus_tag: str,
    focus_pockets: set[str],
    range_active: bool,
    range_soft_active: bool,
    range_ctx,
    event_soon: bool,
    spread_gate_active: bool,
    spread_gate_reason: str,
    spread_log_context: str,
    lot_allocation: float,
    risk_override: float,
    weight_macro: float,
    scalp_share: float,
    signals: list[dict],
    perf_cache: dict,
    fac_m1: dict,
    fac_h4: dict,
    notes: dict | None = None,
    spread_macro_relaxed: bool = False,
    spread_micro_relaxed: bool = False,
) -> PocketPlan:
    range_ctx_info = {
        "active": getattr(range_ctx, "active", False),
        "score": getattr(range_ctx, "score", 0.0),
        "reason": getattr(range_ctx, "reason", ""),
        "metrics": getattr(range_ctx, "metrics", {}),
    }
    plan_notes = dict(notes or {})
    plan_notes.setdefault("range_reason", range_ctx_info["reason"])
    plan_notes.setdefault("range_score", range_ctx_info["score"])
    plan_notes.setdefault("usd_long_cap_lot", _EXPOSURE_USD_LONG_MAX_LOT)
    plan_notes.setdefault("spread_macro_relaxed", spread_macro_relaxed)
    plan_notes.setdefault("spread_micro_relaxed", spread_micro_relaxed)
    factors_m1_view = {
        k: v for k, v in (fac_m1 or {}).items() if k != "candles"
    }
    factors_h4_view = {
        k: v for k, v in (fac_h4 or {}).items() if k != "candles"
    }
    return PocketPlan(
        generated_at=now,
        pocket=pocket,  # type: ignore[arg-type]
        focus_tag=focus_tag,
        focus_pockets=sorted(focus_pockets),
        range_active=range_active,
        range_soft_active=range_soft_active,
        range_ctx=range_ctx_info,
        event_soon=event_soon,
        spread_gate_active=spread_gate_active,
        spread_gate_reason=spread_gate_reason,
        spread_log_context=spread_log_context,
        lot_allocation=round(float(lot_allocation or 0.0), 4),
        risk_override=float(risk_override or 0.0),
        weight_macro=float(weight_macro or 0.0),
        scalp_share=float(scalp_share or 0.0),
        signals=list(signals or []),
        perf_snapshot=dict(perf_cache or {}),
        factors_m1=factors_m1_view,
        factors_h4=factors_h4_view,
        notes=plan_notes,
    )


def _publish_pocket_plans(
    *,
    now: datetime.datetime,
    focus_tag: str,
    focus_pockets: set[str],
    range_active: bool,
    range_soft_active: bool,
    range_ctx,
    event_soon: bool,
    spread_gate_active: bool,
    spread_gate_reason: str,
    spread_log_context: str,
    lots: dict[str, float],
    risk_override: float,
    weight_macro: float,
    scalp_share: float,
    evaluated_signals: list[dict],
    perf_cache: dict,
    fac_m1: dict,
    fac_h4: dict,
    notes: dict[str, float],
    spread_macro_relaxed: bool = False,
    spread_micro_relaxed: bool = False,
) -> None:
    for pocket in ("macro", "scalp"):
        if not _pocket_worker_owns_orders(pocket):
            continue
        plan_signals = [
            dict(sig) for sig in evaluated_signals if sig.get("pocket") == pocket
        ]
        plan = _build_pocket_plan(
            now=now,
            pocket=pocket,
            focus_tag=focus_tag,
            focus_pockets=focus_pockets,
            range_active=range_active,
            range_soft_active=range_soft_active,
            range_ctx=range_ctx,
            event_soon=event_soon,
            spread_gate_active=spread_gate_active,
            spread_gate_reason=spread_gate_reason,
            spread_log_context=spread_log_context,
            lot_allocation=lots.get(pocket, 0.0),
            risk_override=risk_override,
            weight_macro=weight_macro,
            scalp_share=scalp_share,
            signals=plan_signals,
            perf_cache=perf_cache,
            fac_m1=fac_m1,
            fac_h4=fac_h4,
            notes=notes,
            spread_macro_relaxed=spread_macro_relaxed if pocket == "macro" else False,
            spread_micro_relaxed=spread_micro_relaxed if pocket != "macro" else False,
        )
        plan_bus.publish(plan)


_ORDERS_DB_PATH = Path("logs/orders.db")
_TRADES_DB_PATH = Path("logs/trades.db")


def _parse_trade_ts(value: Optional[str]) -> datetime.datetime:
    if not value:
        return datetime.datetime.utcnow()
    try:
        ts = value.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(ts)
    except ValueError:
        return datetime.datetime.utcnow()
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc)
        dt = dt.replace(tzinfo=None)
    return dt


def _lookup_client_id(ticket_id: str) -> Optional[str]:
    if not ticket_id or not _ORDERS_DB_PATH.exists():
        return None
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect(_ORDERS_DB_PATH)
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT client_order_id
            FROM orders
            WHERE ticket_id = ?
              AND status = 'filled'
            ORDER BY id ASC
            LIMIT 1
            """,
            (ticket_id,),
        ).fetchone()
    except sqlite3.Error as exc:
        logging.debug("[ORDERS] lookup failed for %s: %s", ticket_id, exc)
        return None
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
    if not row:
        return None
    client_id = row["client_order_id"] if "client_order_id" in row.keys() else row[0]
    return client_id or None


def _latest_trade_entry_time(trades_db: Path = _TRADES_DB_PATH) -> datetime.datetime | None:
    if not trades_db.exists():
        return None
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect(trades_db)
        row = con.execute(
            "SELECT entry_time FROM trades WHERE entry_time IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
    except sqlite3.Error as exc:
        logging.debug("[IDLE] last trade lookup failed: %s", exc)
        return None
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
    if not row:
        return None
    dt = _parse_iso8601(str(row[0])) if row[0] else None
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt


def _trendma_signal_allowed(
    raw_signal: Dict,
    meta: Dict,
    now: datetime.datetime,
    last_exit: Optional[Dict[str, object]],
) -> bool:
    action = raw_signal.get("action")
    if action not in {"OPEN_LONG", "OPEN_SHORT"}:
        return True
    direction = "long" if action == "OPEN_LONG" else "short"
    if last_exit and last_exit.get("side") and last_exit["side"] != direction:
        since_exit = max(0.0, (now - last_exit["time"]).total_seconds())
        if since_exit < TRENDMA_FLIP_LOOKBACK_SEC:
            strength_ratio = meta.get("strength_ratio")
            if strength_ratio is not None and strength_ratio < TRENDMA_FLIP_STRENGTH_RATIO:
                logging.info(
                    "[TRENDMA] skip flip %s (strength %.2f < %.2f, %.0fs since exit)",
                    direction,
                    strength_ratio,
                    TRENDMA_FLIP_STRENGTH_RATIO,
                    since_exit,
                )
                return False
            slope = abs(meta.get("gap_slope_pips") or 0.0)
            if slope < TRENDMA_FLIP_SLOPE_MIN:
                logging.info(
                    "[TRENDMA] skip flip %s (slope %.3f < %.3f, %.0fs since exit)",
                    direction,
                    slope,
                    TRENDMA_FLIP_SLOPE_MIN,
                    since_exit,
                )
                return False
            adx_val = meta.get("adx")
            if adx_val is not None and adx_val < TRENDMA_FLIP_ADX_MIN:
                logging.info(
                    "[TRENDMA] skip flip %s (ADX %.1f < %.1f, %.0fs since exit)",
                    direction,
                    adx_val,
                    TRENDMA_FLIP_ADX_MIN,
                    since_exit,
                )
                return False
    return True


def _apply_trade_cooldowns(
    trades: list,
    stage_tracker: StageTracker,
    now: datetime.datetime,
) -> Tuple[Optional[datetime.datetime], Optional[Dict[str, object]]]:
    freeze_until: Optional[datetime.datetime] = None
    last_exit: Optional[Dict[str, object]] = None
    for record in trades or []:
        if not isinstance(record, dict):
            continue
        pocket = record.get("pocket") or ""
        ticket_id = str(record.get("ticket_id") or "")
        try:
            units_val = int(record.get("units") or 0)
        except (TypeError, ValueError):
            units_val = 0
        if not ticket_id:
            continue
        if units_val == 0:
            continue
        client_id = _lookup_client_id(ticket_id)
        if client_id and "NewsSpike" in client_id:
            candidate = now + datetime.timedelta(seconds=TRENDMA_NEWS_FREEZE_SECONDS)
            if not freeze_until or candidate > freeze_until:
                freeze_until = candidate
        if not client_id or "TrendMA" not in client_id:
            continue
        entry_ts = _parse_trade_ts(record.get("entry_time"))
        close_ts = _parse_trade_ts(record.get("close_time"))
        hold_seconds = max(0.0, (close_ts - entry_ts).total_seconds())
        pnl = float(record.get("pl_pips") or 0.0)
        direction = "long" if units_val > 0 else "short"
        last_exit = {
            "side": direction,
            "time": close_ts,
            "pnl": pnl,
            "hold_seconds": hold_seconds,
        }
        if pocket != "macro":
            continue
        if pnl < 0 and hold_seconds < TRENDMA_FAST_LOSS_THRESHOLD_SEC:
            lose_streak, _ = stage_tracker.get_loss_profile(pocket, direction)
            extra_seconds = TRENDMA_FAST_LOSS_BASE_COOLDOWN + max(0, lose_streak) * TRENDMA_FAST_LOSS_STREAK_STEP
            applied = stage_tracker.ensure_cooldown(
                pocket,
                direction,
                reason="fast_loss",
                seconds=extra_seconds,
                now=now,
            )
            if applied:
                logging.info(
                    "[TRENDMA] Fast-loss cooldown applied pocket=%s dir=%s sec=%d streak=%d",
                    pocket,
                    direction,
                    extra_seconds,
                    lose_streak,
                )
            opp_dir = "short" if direction == "long" else "long"
            opp_seconds = max(240, extra_seconds // 2)
            stage_tracker.ensure_cooldown(
                pocket,
                opp_dir,
                reason="flip_guard_fast_loss",
                seconds=opp_seconds,
                now=now,
            )
    return freeze_until, last_exit


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


async def m1_candle_handler(cndl: Candle):
    await on_candle("M1", cndl)


async def h1_candle_handler(cndl: Candle):
    await on_candle("H1", cndl)


async def h4_candle_handler(cndl: Candle):
    await on_candle("H4", cndl)


async def logic_loop():
    global _MANUAL_SENTINEL_ACTIVE, _MANUAL_SENTINEL_CLEAR_STREAK, _HOLD_RATIO_GUARD_ACTIVE
    pos_manager = PositionManager()
    metrics_client = RealtimeMetricsClient()
    confidence_policy = ConfidencePolicy()
    exit_manager = ExitManager()
    stage_tracker = StageTracker()
    pattern_stats = PatternStats()
    factor_warmup_state: Optional[dict] = None
    perf_cache = {}
    news_cache = {}
    insight = InsightClient()
    missing_factor_cycles = 0
    last_update_time = datetime.datetime.min
    last_heartbeat_time = datetime.datetime.min  # Add this line
    last_metrics_refresh = datetime.datetime.min
    last_macro_snapshot_refresh = datetime.datetime.min
    last_trade_check_time = datetime.datetime.min
    last_trade_entry: Optional[datetime.datetime] = _latest_trade_entry_time()
    idle_refresh_notified = False
    idle_refresh_last = datetime.datetime.min
    strategy_health_cache: dict[str, StrategyHealth] = {}
    range_active = False
    range_soft_active = False
    last_range_reason = ""
    range_state_since = datetime.datetime.min
    range_entry_counter = 0
    range_exit_counter = 0
    raw_range_active = False
    raw_range_reason = ""
    range_breakout_release_until = datetime.datetime.min
    range_breakout_reason = ""
    range_override_active_prev = False
    last_range_scalp_ready: Optional[bool] = None
    range_macro_hold_until = datetime.datetime.min
    stage_empty_since: dict[tuple[str, str], datetime.datetime] = {}
    last_risk_pct: float | None = None
    last_spread_gate = False
    last_spread_gate_reason = ""
    market_closed_logged = False
    last_market_closed_log = datetime.datetime.min
    trendma_news_cooldown_until = datetime.datetime.min
    trendma_last_exit: Optional[Dict[str, object]] = None
    last_hold_ratio_check = datetime.datetime.min
    price_window: deque[tuple[datetime.datetime, float]] = deque(maxlen=128)
    last_surge_trigger = datetime.datetime.min
    start_time = datetime.datetime.utcnow()
    cold_start_logged = False

    try:
        while True:
            now = datetime.datetime.utcnow()
            elapsed_since_start = (now - start_time).total_seconds()
            if elapsed_since_start < COLD_START_GRACE_SEC:
                if not cold_start_logged:
                    logging.info(
                        "[COLD-START] Grace period %.0fs (elapsed %.0fs); skipping entries.",
                        COLD_START_GRACE_SEC,
                        elapsed_since_start,
                    )
                    cold_start_logged = True
                await asyncio.sleep(5)
                continue
            stage_tracker.clear_expired(now)
            recent_profiles: dict[str, dict[str, float]] = {}
            try:
                stage_tracker.update_loss_streaks(now=now, cooldown_map=POCKET_LOSS_COOLDOWNS)
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("[STAGE] loss streak update failed: %s", exc)
            else:
                recent_profiles = stage_tracker.recent_profiles
            if not recent_profiles:
                recent_profiles = stage_tracker.recent_profiles
            if (now - last_trade_check_time).total_seconds() >= IDLE_REFRESH_CHECK_SEC:
                last_trade_check_time = now
                try:
                    latest_entry = _latest_trade_entry_time()
                except Exception as exc:  # pragma: no cover - defensive
                    logging.debug("[IDLE] trade lookup failed: %s", exc)
                    latest_entry = None
                if latest_entry:
                    last_trade_entry = latest_entry
                idle_gap = (
                    (now - last_trade_entry).total_seconds()
                    if last_trade_entry is not None
                    else None
                )
                if idle_gap is not None and idle_gap >= IDLE_REFRESH_THRESHOLD_SEC:
                    if not idle_refresh_notified:
                        logging.warning(
                            "[IDLE] No trades for %.1f minutes; refreshing factors and macro snapshot.",
                            idle_gap / 60.0,
                        )
                        idle_refresh_notified = True
                    if (now - idle_refresh_last).total_seconds() >= max(60.0, IDLE_REFRESH_CHECK_SEC):
                        try:
                            await initialize_history(TARGET_INSTRUMENT)
                        except Exception as exc:  # pragma: no cover - defensive
                            logging.warning("[IDLE] history refresh failed: %s", exc)
                        try:
                            await asyncio.to_thread(
                                refresh_macro_snapshot,
                                snapshot_path=_macro_snapshot_path(),
                                deadzone=_MACRO_STATE_DEADZONE,
                                refresh_if_older_than_minutes=int(_MACRO_SNAPSHOT_REFRESH_MINUTES),
                            )
                            globals()["_macro_state_cache"] = None
                            _refresh_macro_state()
                        except Exception as exc:  # pragma: no cover - defensive
                            logging.warning("[IDLE] macro snapshot refresh failed: %s", exc)
                        idle_refresh_last = now
                else:
                    if idle_refresh_notified and idle_gap is not None:
                        logging.info(
                            "[IDLE] Activity detected after %.1f minutes gap.",
                            idle_gap / 60.0,
                        )
                    idle_refresh_notified = False

            if (now - last_hold_ratio_check).total_seconds() >= HOLD_RATIO_CHECK_INTERVAL_SEC:
                ratio, total, lt60 = _HOLD_MONITOR.sample()
                last_hold_ratio_check = now
                if ratio is not None:
                    if (not _HOLD_RATIO_GUARD_ACTIVE) and ratio > HOLD_RATIO_MAX:
                        _HOLD_RATIO_GUARD_ACTIVE = True
                        log_metric(
                            "hold_ratio_guard",
                            1.0,
                            tags={"ratio": f"{ratio:.3f}", "samples": str(total)},
                        )
                        logging.warning(
                            "[HOLD] ratio guard activated (ratio=%.1f%%, total=%s, lt60=%s)",
                            ratio * 100,
                            total,
                            lt60,
                        )
                    elif _HOLD_RATIO_GUARD_ACTIVE and ratio < HOLD_RATIO_MAX * HOLD_RATIO_RELEASE_FACTOR:
                        _HOLD_RATIO_GUARD_ACTIVE = False
                        log_metric(
                            "hold_ratio_guard",
                            0.0,
                            tags={"ratio": f"{ratio:.3f}", "samples": str(total)},
                        )
                        logging.info(
                            "[HOLD] ratio guard released (ratio=%.1f%%, total=%s)",
                            ratio * 100,
                            total,
                        )

            if factor_warmup_state is not None:
                try:
                    await ensure_factor_history_ready(factor_warmup_state)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.exception("[FACTOR] warmup check failed; continuing")
            stage_tracker.clear_expired(now)
            # Heartbeat logging
            if (now - last_heartbeat_time).total_seconds() >= 300:  # Every 5 minutes
                logging.info(
                    f"[HEARTBEAT] System is alive at {now.isoformat(timespec='seconds')}"
                )
                last_heartbeat_time = now

            # 5分ごとにパフォーマンスとニュースを更新
            if (now - last_update_time).total_seconds() >= 300:
                perf_cache = get_perf()
                news_cache = get_latest_news(
                    limit_short=NEWS_LIMITS["short"],
                    limit_long=NEWS_LIMITS["long"],
                )
                try:
                    insight.refresh()
                except Exception:
                    pass
                try:
                    pattern_stats.refresh(now=now)
                except Exception as exc:
                    logging.warning("[PATTERN] refresh failed: %s", exc)
                last_update_time = now
                logging.info(f"[PERF] Updated: {perf_cache}")
                logging.info(f"[NEWS] Updated: {news_cache}")

            # --- 1. 状況分析 ---
            factors = all_factors()
            fac_m1 = dict(factors.get("M1") or {})
            fac_h4 = dict(factors.get("H4") or {})

            # 両方のタイムフレームのデータが揃うまで待機
            if (
                not fac_m1
                or not fac_h4
                or not fac_m1.get("close")
                or not fac_h4.get("close")
            ):
                missing_factor_cycles += 1
                if missing_factor_cycles % 12 == 0:
                    logging.warning(
                        "[WAIT] Factor data unavailable for %d cycles; reloading history.",
                        missing_factor_cycles,
                    )
                    try:
                        await initialize_history(TARGET_INSTRUMENT)
                    except Exception as exc:  # pragma: no cover - defensive
                        logging.warning("[WAIT] initialize_history retry failed: %s", exc)
                else:
                    logging.info("[WAIT] Waiting for M1/H4 factor data for trading logic...")
                await asyncio.sleep(5)
                continue
            if missing_factor_cycles:
                logging.info(
                    "[WAIT] Factor data restored after %d cycles.", missing_factor_cycles
                )
                missing_factor_cycles = 0

            if (now - last_macro_snapshot_refresh).total_seconds() >= 900:
                try:
                    await asyncio.to_thread(
                        refresh_macro_snapshot,
                        snapshot_path=_macro_snapshot_path(),
                        deadzone=_MACRO_STATE_DEADZONE,
                        refresh_if_older_than_minutes=int(_MACRO_SNAPSHOT_REFRESH_MINUTES),
                    )
                    last_macro_snapshot_refresh = now
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("[MACRO] snapshot rebuild failed: %s", exc)

            macro_state = _refresh_macro_state()
            macro_snapshot_stale = False
            macro_snapshot_age = 0.0
            event_soon = check_event_soon(within_minutes=30, min_impact=3)

            if macro_state and _MACRO_STATE_STALE_WARN_SEC > 0:
                snap_ts = _parse_iso8601(macro_state.snapshot.asof)
                if snap_ts:
                    age_sec = (
                        now.replace(tzinfo=datetime.timezone.utc) - snap_ts
                    ).total_seconds()
                    macro_snapshot_age = age_sec
                    if age_sec >= _MACRO_STATE_STALE_WARN_SEC:
                        macro_snapshot_stale = True
                        if not _macro_state_stale_warned:
                            logging.warning(
                                "[MACRO] snapshot stale (age=%.1fs threshold=%.1fs)",
                                age_sec,
                                _MACRO_STATE_STALE_WARN_SEC,
                            )
                            globals()["_macro_state_stale_warned"] = True
                        if _MACRO_AUTO_REFRESH_ON_STALE:
                            try:
                                await asyncio.to_thread(
                                    refresh_macro_snapshot,
                                    snapshot_path=_macro_snapshot_path(),
                                    deadzone=_MACRO_STATE_DEADZONE,
                                    refresh_if_older_than_minutes=int(_MACRO_SNAPSHOT_REFRESH_MINUTES),
                                )
                                last_macro_snapshot_refresh = now
                                _macro_state_cache = None
                                macro_state = _refresh_macro_state()
                                # リフレッシュ成功なら縮退を解除
                                macro_snapshot_stale = False
                                macro_snapshot_age = 0.0
                                logging.info(
                                    "[MACRO] snapshot refreshed on stale detect (age=%.1fs)",
                                    age_sec,
                                )
                            except Exception as exc:  # pragma: no cover
                                logging.warning("[MACRO] auto refresh failed: %s", exc)
                    elif _macro_state_stale_warned:
                        logging.info("[MACRO] snapshot freshness restored (age=%.1fs)", age_sec)
                        globals()["_macro_state_stale_warned"] = False
            global_drawdown_exceeded = check_global_drawdown()

            if global_drawdown_exceeded:
                logging.warning(
                    "[STOP] Global drawdown limit exceeded. Stopping new trades."
                )
                await asyncio.sleep(60)
                continue

            spread_blocked, spread_remain, spread_snapshot, spread_reason = spread_monitor.is_blocked()
            spread_live_pips: Optional[float] = None
            if spread_snapshot and spread_snapshot.get("spread_pips") is not None:
                try:
                    spread_live_pips = float(spread_snapshot.get("spread_pips"))
                except (TypeError, ValueError):
                    spread_live_pips = None
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
            spread_macro_relaxed = (
                spread_gate_active
                and spread_live_pips is not None
                and spread_live_pips <= SPREAD_OVERRIDE_MACRO_PIPS
            )
            spread_micro_relaxed = (
                spread_gate_active
                and spread_live_pips is not None
                and spread_live_pips <= SPREAD_OVERRIDE_MICRO_PIPS
            )
            if spread_macro_relaxed and spread_gate_active:
                logging.info(
                    "[SPREAD] macro override active (%.2fp <= %.2fp)",
                    spread_live_pips or -1.0,
                    SPREAD_OVERRIDE_MACRO_PIPS,
                )
            if spread_micro_relaxed and spread_gate_active:
                logging.info(
                    "[SPREAD] micro override active (%.2fp <= %.2fp)",
                    spread_live_pips or -1.0,
                    SPREAD_OVERRIDE_MICRO_PIPS,
                )
            if spread_live_pips is not None:
                fac_m1["spread_pips"] = spread_live_pips
                fac_h4["spread_pips"] = spread_live_pips
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

            # --- Market hours gate (skip full cycle when closed) ---
            if not is_market_open(now):
                if (
                    not market_closed_logged
                    or (now - last_market_closed_log).total_seconds() >= 600
                ):
                    logging.info(
                        "[MARKET] Closed (UTC=%s). Sleeping until reopen window.",
                        now.isoformat(timespec="seconds"),
                    )
                    last_market_closed_log = now
                    market_closed_logged = True
                await asyncio.sleep(30)
                continue
            market_closed_logged = False

            # --- 2. GPT判断 ---
            if FORCE_SCALP_MODE:
                logging.warning("[FORCE_SCALP] entering GPT stage loop=%d", loop_counter)
            # M1/H4 の移動平均・RSI などの指標をまとめて送信
            news_features = build_news_features(news_cache, now=now)
            news_status, news_age_min, news_count_total, news_impact_max = _resolve_news_status(news_features)
            low_vol_ctx = _low_vol_profile(fac_m1, fac_h4)
            low_vol_active = bool(low_vol_ctx["low_vol"])
            low_vol_like = bool(low_vol_ctx["low_vol_like"])
            quiet_low_vol = bool(low_vol_ctx["tight_vol"]) and news_status in {"quiet", "stale"}
            low_vol_enabled = _env_flag("LOWVOL_ENABLE", True)
            canary_ok = low_vol_enabled
            if low_vol_enabled:
                canary_symbols = set(_env_csv("LOWVOL_CANAIRY_SYMBOLS"))
                if canary_symbols and PRIMARY_SYMBOL not in canary_symbols:
                    canary_ok = False
                if canary_ok:
                    hours_spec = os.getenv("LOWVOL_CANAIRY_HOURS")
                    if hours_spec and not _within_canary_hours(hours_spec, now):
                        canary_ok = False
            if not canary_ok:
                low_vol_active = False
                low_vol_like = False
                quiet_low_vol = False
            apply_low_vol = low_vol_enabled and canary_ok
            if apply_low_vol and (low_vol_active or quiet_low_vol):
                log_metric(
                    "low_volatility_score",
                    float(low_vol_ctx["score"]),
                    tags={
                        "news_status": news_status,
                        "quiet": str(quiet_low_vol).lower(),
                    },
                    ts=now,
                )
            payload = {
                "ts": now.isoformat(timespec="seconds"),
                "reg_macro": macro_regime,
                "reg_micro": micro_regime,
                "factors_m1": _compact_factors(fac_m1, GPT_FACTOR_KEYS["M1"]),
                "factors_h1": _compact_factors(fac_h1, GPT_FACTOR_KEYS["H1"]),
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
                "news_features": news_features,
                "event_soon": event_soon,
            }
            # GPT判断（フォールバックなし）。失敗時はこのループをスキップ。
            try:
                gpt = await get_decision(payload)
            except Exception as e:
                logging.warning(f"[SKIP] GPT decision unavailable: {e}")
                await asyncio.sleep(5)
                continue
            gpt_strategies_raw = list(gpt.get("ranked_strategies", []))
            logging.info(
                "[GPT] focus=%s weight_macro=%.2f strategies=%s",
                gpt.get("focus_tag"),
                gpt.get("weight_macro", 0.0),
                gpt_strategies_raw,
            )
            micro_gpt_hint = [
                s
                for s in gpt_strategies_raw
                if STRATEGIES.get(s) and STRATEGIES[s].pocket == "micro"
            ]
            ranked_strategies = list(gpt_strategies_raw)
            if _DELEGATE_MICRO:
                ranked_strategies = [
                    s
                    for s in ranked_strategies
                    if STRATEGIES.get(s) and STRATEGIES[s].pocket != "micro"
                ]

            # Update realtime metrics cache every few minutes
            if (now - last_metrics_refresh).total_seconds() >= 240:
                try:
                    metrics_client.refresh()
                    strategy_health_cache.clear()
                    last_metrics_refresh = now
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("[REALTIME] metrics refresh failed: %s", exc)

            _maybe_run_online_tuner(now)

            atr_pips = (fac_m1.get("atr") or 0.0) * 100
            ema20 = fac_m1.get("ema20") or fac_m1.get("ma20")
            close_px = fac_m1.get("close")
            momentum = 0.0
            if ema20 is not None and close_px is not None:
                momentum = close_px - ema20
            surge_triggered = False
            surge_move = None
            surge_long_move = None
            if close_px is not None:
                price_window.append((now, close_px))
                cutoff_short = now - datetime.timedelta(seconds=PRICE_SURGE_WINDOW_SEC)
                cutoff_long = now - datetime.timedelta(seconds=PRICE_SURGE_LONG_WINDOW_SEC)
                window_cap = max(PRICE_SURGE_LONG_WINDOW_SEC, PRICE_SURGE_WINDOW_SEC) * 2
                while price_window and (now - price_window[0][0]).total_seconds() > window_cap:
                    price_window.popleft()
                # 最も古い価格（短窓/長窓）を使って変化量を算出
                short_px = None
                long_px = None
                for ts, px in price_window:
                    if ts <= cutoff_long and long_px is None:
                        long_px = px
                    if ts <= cutoff_short and short_px is None:
                        short_px = px
                if short_px is not None:
                    surge_move = close_px - short_px
                if long_px is not None:
                    surge_long_move = close_px - long_px
            vol_recent = 0.0
            try:
                vol_raw = fac_m1.get("vol_5m")
                if vol_raw is not None:
                    vol_recent = float(vol_raw)
            except (TypeError, ValueError):
                vol_recent = 0.0
            surge_hit = False
            surge_val = None
            surge_window = None
            if surge_move is not None and abs(surge_move) >= PRICE_SURGE_MIN_MOVE:
                surge_hit = True
                surge_val = surge_move
                surge_window = PRICE_SURGE_WINDOW_SEC
            if (
                surge_long_move is not None
                and abs(surge_long_move) >= PRICE_SURGE_LONG_MIN_MOVE
            ):
                # 長窓の方が大きければそちらを優先
                if not surge_hit or abs(surge_long_move) > abs(surge_move or 0.0):
                    surge_hit = True
                    surge_val = surge_long_move
                    surge_window = PRICE_SURGE_LONG_WINDOW_SEC
            if (
                surge_hit
                and atr_pips >= PRICE_SURGE_ATR_MIN
                and vol_recent >= PRICE_SURGE_VOL_MIN
                and (now - last_surge_trigger).total_seconds() >= PRICE_SURGE_COOLDOWN_SEC
            ):
                surge_triggered = True
                last_surge_trigger = now
                surge_dir = "long" if surge_val > 0 else "short"
                logging.info(
                    "[SURGE] %.0fs move %.3f (ATR %.2f vol5m %.2f)",
                    surge_window or PRICE_SURGE_WINDOW_SEC,
                    surge_val,
                    atr_pips,
                    vol_recent,
                )
                try:
                    for pocket in ("macro", "micro"):
                        applied = stage_tracker.ensure_cooldown(
                            pocket,
                            surge_dir,
                            reason="surge_guard",
                            seconds=PRICE_SURGE_BLOCK_SEC,
                            now=now,
                        )
                        if applied:
                            logging.info(
                                "[SURGE] Block %s %s for %ss after %.2fp move.",
                                pocket,
                                surge_dir,
                                PRICE_SURGE_BLOCK_SEC,
                                surge_val / 0.01,
                            )
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("[SURGE] cooldown set failed: %s", exc)
            scalp_ready = False
            scalp_atr_min = RANGE_SCALP_ATR_MIN if range_active else 2.2
            scalp_momentum_min = RANGE_SCALP_MOMENTUM_MIN if range_active else 0.0015
            if atr_pips >= scalp_atr_min and abs(momentum) >= scalp_momentum_min:
                scalp_ready = True
            elif fac_m1.get("vol_5m"):
                vol_floor = RANGE_SCALP_VOL_MIN if range_active else 1.2
                atr_floor = RANGE_SCALP_ATR_MIN if range_active else 2.0
                scalp_ready = atr_pips >= atr_floor and fac_m1["vol_5m"] >= vol_floor

            focus_tag = gpt.get("focus_tag") or focus
            weight = gpt.get("weight_macro", w_macro)
            cap = _macro_weight_cap(perf_cache)
            if weight > cap:
                logging.info(
                    "[MACRO] Performance cap applied: weight_macro %.2f -> %.2f (win_rate=%s)",
                    weight,
                    cap,
                    (perf_cache.get("macro") or {}).get("win_rate"),
                )
                weight = cap
            if macro_snapshot_stale:
                if focus_tag == "macro":
                    focus_tag = "hybrid"
                prev_weight = weight
                weight = min(prev_weight * _MACRO_STALE_WEIGHT_DECAY, _MACRO_STALE_WEIGHT_CAP)
                logging.warning(
                    "[MACRO] Snapshot stale (age=%.1fs). weight_macro %.2f -> %.2f and focus=%s",
                    macro_snapshot_age,
                    prev_weight,
                    weight,
                    focus_tag,
                )
            if weight > GLOBAL_MACRO_WEIGHT_CAP:
                logging.info(
                    "[MACRO] Hard cap applied: weight_macro %.2f -> %.2f",
                    weight,
                    GLOBAL_MACRO_WEIGHT_CAP,
                )
                weight = GLOBAL_MACRO_WEIGHT_CAP
            if range_active:
                if focus_tag == "macro":
                    focus_tag = "hybrid"
                prev_weight = weight
                weight = min(weight, RANGE_MACRO_WEIGHT_CAP)
                if prev_weight != weight:
                    logging.info(
                        "[MACRO] Range compression cap applied: weight_macro %.2f -> %.2f",
                        prev_weight,
                        weight,
                    )
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
            focus_pockets = set(FOCUS_POCKETS.get(focus_tag, ("macro", "micro", "scalp")))
            if (
                macro_snapshot_stale
                and macro_snapshot_age >= MACRO_STALE_DISABLE_SEC
                and "macro" in focus_pockets
            ):
                focus_pockets.discard("macro")
                logging.info(
                    "[MACRO] Disabled macro pocket until snapshot refresh (age=%.1fs).",
                    macro_snapshot_age,
                )
            if not focus_pockets:
                focus_pockets = {"micro"}

            if surge_triggered:
                momentum_added = False
                if _DELEGATE_MICRO:
                    if "MomentumBurst" not in micro_gpt_hint:
                        micro_gpt_hint.append("MomentumBurst")
                        momentum_added = True
                else:
                    if "MomentumBurst" not in ranked_strategies:
                        ranked_strategies.append("MomentumBurst")
                        momentum_added = True
                if momentum_added:
                    logging.info("[SURGE] MomentumBurst boost via 10m move.")
                if "ImpulseRetrace" not in ranked_strategies:
                    ranked_strategies.append("ImpulseRetrace")
                    logging.info("[SURGE] Added ImpulseRetrace due to surge.")
                if "scalp" in focus_pockets:
                    scalp_ready = True

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
            if os.getenv("SCALP_TACTICAL", "0").strip().lower() not in {"","0","false","no"}:
                focus_pockets.add("scalp")
                if "M1Scalper" not in ranked_strategies:
                    ranked_strategies.append("M1Scalper")
                if "BB_RSI" not in ranked_strategies:
                    ranked_strategies.append("BB_RSI")

                logging.info(
                    "[SCALP] Auto-added M1Scalper (range=%s ATR %.2f, momentum %.4f, vol5m %.2f).",
                    range_active,
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
                logging.info(
                    "[SCALP-MAIN] Range mode: auto-added RangeFader (score=%.2f bbw=%.2f atr=%.2f).",
                    range_ctx.score,
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
                logging.info(
                    "[SCALP-MAIN] Auto-added PulseBreak (mom=%.4f atr=%.2f vol5m=%.2f).",
                    momentum,
                    atr_pips,
                    vol_5m,
                )

            evaluated_signals: list[dict] = []
            strategies_by_pocket: dict[str, list[str]] = defaultdict(list)
            if _DELEGATE_MICRO and micro_gpt_hint:
                strategies_by_pocket["micro"].extend(micro_gpt_hint)
            for sname in ranked_strategies:
                cls = STRATEGIES.get(sname)
                if not cls:
                    continue
                pocket = cls.pocket
                if (
                    (_MANUAL_SENTINEL_ACTIVE and pocket in MANUAL_SENTINEL_BLOCK_POCKETS)
                    or (_HOLD_RATIO_GUARD_ACTIVE and pocket in HOLD_RATIO_BLOCK_POCKETS)
                ):
                    reason = (
                        "manual exposure"
                        if _MANUAL_SENTINEL_ACTIVE and pocket in MANUAL_SENTINEL_BLOCK_POCKETS
                        else "hold_ratio_guard"
                    )
                    logging.info(
                        "[GUARD] skip %s pocket=%s reason=%s",
                        sname,
                        pocket,
                        reason,
                    )
                    continue
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
                strategies_by_pocket[pocket].append(sname)
                if pocket == "micro" and _DELEGATE_MICRO:
                    continue
                if sname == "NewsSpikeReversal":
                    raw_signal = cls.check(fac_m1, news_cache.get("short", []))
                else:
                    # Use H4 factors for macro strategies to enforce H4 perspective,
                    # but pass lower-TF aggregates (H1/M10/M5) for prediction confluence.
                    if pocket == "macro":
                        try:
                            from analysis.mtf_utils import resample_candles_from_m1
                            c_m1 = fac_m1.get("candles") or []
                            # H1 comes from factor cache if available; otherwise build from M1
                            factors_all = all_factors()
                            fac_h1 = factors_all.get("H1") or {}
                            c_h1 = fac_h1.get("candles") or resample_candles_from_m1(c_m1, 60)
                            c_m10 = resample_candles_from_m1(c_m1, 10)
                            c_m5 = resample_candles_from_m1(c_m1, 5)
                            fac_h4_mtf = dict(fac_h4)
                            fac_h4_mtf["mtf"] = {
                                "candles_h1": c_h1,
                                "candles_m10": c_m10,
                                "candles_m5": c_m5,
                            }
                            raw_signal = cls.check(fac_h4_mtf)
                        except Exception:
                            raw_signal = cls.check(fac_h4)
                    else:
                        raw_signal = cls.check(fac_m1)
                if not raw_signal:
                    if FORCE_SCALP_MODE and pocket == "scalp":
                        logging.warning("[FORCE_SCALP] %s returned None", sname)
                    continue
                meta = raw_signal.pop("_meta", None)
                if sname == "TrendMA":
                    if now < trendma_news_cooldown_until:
                        remaining = (trendma_news_cooldown_until - now).total_seconds()
                        logging.info(
                            "[TRENDMA] skip TrendMA during news freeze (%.0fs remaining).",
                            max(1, remaining),
                        )
                        continue
                    if not _trendma_signal_allowed(raw_signal, meta or {}, now, trendma_last_exit):
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
                    "profile": _extract_profile_name(raw_signal, getattr(cls, "name", sname)),
                }
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
                if (
                    range_active
                    and range_macro_bias_dir
                    and signal["pocket"] == "macro"
                    and cls.name in ALLOWED_RANGE_MACRO_STRATEGIES
                ):
                    notes = signal.get("notes")
                    if not isinstance(notes, dict):
                        notes = {}
                    notes.setdefault("range_bias_dir", range_macro_bias_dir)
                    notes.setdefault("range_bias_source", cls.name)
                    signal["notes"] = notes
                    signal["range_bias_dir"] = range_macro_bias_dir
                scaled_conf = int(signal["confidence"] * health.confidence_scale)
                signal["confidence"] = max(0, min(100, scaled_conf))
                if (
                    _MACRO_STATE_GATE_ENABLED
                    and macro_state
                    and signal["action"] in {"OPEN_LONG", "OPEN_SHORT"}
                ):
                    bias_val = macro_state.bias(TARGET_INSTRUMENT)
                    if abs(bias_val) >= macro_state.deadzone:
                        is_long = signal["action"] == "OPEN_LONG"
                        if (bias_val > 0 and not is_long) or (bias_val < 0 and is_long):
                            logging.info(
                                "[MACRO] gate skip %s action=%s bias=%.2f",
                                sname,
                                signal["action"],
                                bias_val,
                            )
                            continue
                if range_active:
                    atr_hint = (
                        fac_m1.get("atr_pips")
                        or ((fac_m1.get("atr") or 0.0) * 100)
                        or 6.0
                    )
                    pocket = signal["pocket"]
                    if pocket == "macro":
                        tp_cap = min(5.0, max(2.2, atr_hint * 1.4))
                        sl_cap = max(1.8, min(tp_cap * 1.1, atr_hint * 1.6))
                        signal["tp_pips"] = round(tp_cap, 2)
                        signal["sl_pips"] = round(sl_cap, 2)
                    elif pocket == "scalp":
                        tp_existing = signal.get("tp_pips") or max(6.0, atr_hint * 2.8)
                        sl_existing = signal.get("sl_pips") or max(4.5, atr_hint * 1.9)
                        signal["tp_pips"] = round(
                            min(max(5.2, tp_existing), max(8.5, atr_hint * 3.3)), 2
                        )
                        signal["sl_pips"] = round(
                            min(max(3.8, sl_existing), max(6.5, atr_hint * 2.3)), 2
                        )
                    else:
                        tp_default = min(2.4, max(1.4, atr_hint * 1.25))
                        signal["tp_pips"] = round(tp_default, 2)
                        signal["sl_pips"] = round(
                            max(1.2, min(tp_default * 1.08, 2.1)), 2
                        )
                    conf_scale = RANGE_CONFIDENCE_SCALE.get(pocket)
                    if conf_scale is not None:
                        scaled_conf = int(signal["confidence"] * conf_scale)
                        signal["confidence"] = max(15, min(100, scaled_conf))
                    strat_name = signal.get("strategy")
                    damp = RANGE_TREND_CONFIDENCE_DAMP.get(strat_name)
                    if damp is not None:
                        scaled_conf = int(signal["confidence"] * damp)
                        signal["confidence"] = max(15, min(100, scaled_conf))
                elif range_soft_active and pocket in {"macro", "micro"}:
                    # ソフト圧縮時も軽めにタイト化してレンジ捕捉寄りに寄せる
                    atr_hint = (
                        fac_m1.get("atr_pips")
                        or ((fac_m1.get("atr") or 0.0) * 100)
                        or 6.0
                    )
                    tp_existing = signal.get("tp_pips")
                    sl_existing = signal.get("sl_pips")
                    tp_target = tp_existing if tp_existing is not None else max(2.4, atr_hint * 1.4)
                    sl_target = sl_existing if sl_existing is not None else max(1.8, atr_hint * 0.9)
                    signal["tp_pips"] = round(tp_target * 0.9, 2)
                    signal["sl_pips"] = round(max(sl_target * 0.9, 1.2), 2)
                    conf_scale = RANGE_CONFIDENCE_SCALE.get(pocket)
                    if conf_scale is not None:
                        scaled_conf = int(signal["confidence"] * ((conf_scale + 1.0) / 2.0))
                        signal["confidence"] = max(20, min(100, scaled_conf))
                # DISABLE_STOP_LOSS=true のときは broker 側にSLを置かない
                if DISABLE_STOP_LOSS:
                    signal["sl_pips"] = None
                # ATRに応じて micro のSLを底上げし、極端にタイトな初期SLによる即損切りを防ぐ
                if (not DISABLE_STOP_LOSS) and pocket == "micro" and fac_m1.get("atr_pips"):
                    atr_pips = max(0.1, float(fac_m1["atr_pips"]))
                    min_sl = max(5.0, atr_pips * 1.2)
                    if signal.get("sl_pips") is None or signal["sl_pips"] < min_sl:
                        signal["sl_pips"] = round(min_sl, 2)
                signal["min_hold_sec"] = _derive_min_hold_seconds(
                    signal, cls.pocket, fac_m1
                )
                signal["loss_guard_pips"] = _derive_loss_guard_pips(signal, cls.pocket, fac_m1)
                if signal["action"] in {"OPEN_LONG", "OPEN_SHORT"}:
                    pattern_tag, pattern_meta = derive_pattern_signature(
                        fac_m1,
                        action=signal["action"],
                    )
                    if pattern_tag:
                        signal["pattern_tag"] = pattern_tag
                        signal["pattern_meta"] = pattern_meta
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
                            "news_short": (news_cache.get("short", []) if news_cache else [])[:2],
                            "news_long": (news_cache.get("long", []) if news_cache else [])[:1],
                        }
                        rr_hint = await rr_advisor.advise(rr_context)
                    except Exception as exc:  # pragma: no cover - defensive
                        logging.debug("[RR_ADVISOR] failed: %s", exc)
                        rr_hint = None
                    if rr_hint:
                        sl_val = float(signal["sl_pips"] or 0.0)
                        target_tp = round(max(sl_val * rr_hint.ratio, sl_val * rr_advisor.min_ratio), 2)
                        if target_tp > 0.0:
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
                evaluated_signals.append(signal)
                logging.info("[SIGNAL] %s -> %s", cls.name, signal)

            open_positions = pos_manager.get_open_positions()
            # Use filtered positions for bot-controlled actions
            managed_positions = filter_bot_managed_positions(open_positions)
            try:
                resets = stage_tracker.expire_stages_if_flat(
                    open_positions, now=now, grace_seconds=STAGE_RESET_GRACE_SECONDS
                )
                if resets:
                    logging.info(
                        "[STAGE] auto-reset %d stale stages after %.0fs flat",
                        resets,
                        STAGE_RESET_GRACE_SECONDS,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logging.debug("[STAGE] auto-reset failed: %s", exc)
            manual_block_active, manual_units, manual_details = _manual_sentinel_state(
                open_positions
            )
            if manual_block_active:
                _MANUAL_SENTINEL_CLEAR_STREAK = 0
                if not _MANUAL_SENTINEL_ACTIVE:
                    _MANUAL_SENTINEL_ACTIVE = True
                    log_metric(
                        "manual_halt_active",
                        1.0,
                        tags={"units": str(manual_units)},
                    )
                    logging.warning(
                        "[MANUAL] Manual/unknown exposure detected (units=%s detail=%s); "
                        "micro/scalp entries halted.",
                        manual_units,
                        manual_details or "-",
                    )
            else:
                if _MANUAL_SENTINEL_ACTIVE:
                    _MANUAL_SENTINEL_CLEAR_STREAK += 1
                    if _MANUAL_SENTINEL_CLEAR_STREAK >= MANUAL_SENTINEL_RELEASE_CYCLES:
                        _MANUAL_SENTINEL_ACTIVE = False
                        log_metric(
                            "manual_halt_active",
                            0.0,
                            tags={"units": "0"},
                        )
                        logging.info("[MANUAL] Manual/unknown exposure cleared.")
                else:
                    _MANUAL_SENTINEL_CLEAR_STREAK = min(
                        _MANUAL_SENTINEL_CLEAR_STREAK + 1,
                        MANUAL_SENTINEL_RELEASE_CYCLES,
                    )
            managed_for_main = {
                pocket: info
                for pocket, info in managed_positions.items()
                if not _pocket_worker_owns_orders(pocket)
            }
            micro_trades = (managed_positions.get("micro") or {}).get("open_trades", [])
            if any("NewsSpike" in (tr.get("client_id") or "") for tr in micro_trades):
                candidate = now + datetime.timedelta(seconds=TRENDMA_NEWS_FREEZE_SECONDS)
                if candidate > trendma_news_cooldown_until:
                    trendma_news_cooldown_until = candidate
                    logging.info(
                        "[TRENDMA] Active NewsSpike trade detected; freeze TrendMA for %.0fs.",
                        (trendma_news_cooldown_until - now).total_seconds(),
                    )
            try:
                # Only protect bot-managed trades handled by main
                update_dynamic_protections(managed_for_main, fac_m1, fac_h4)
            except Exception as exc:
                logging.warning("[PROTECTION] update failed: %s", exc)

            # Opportunistic macro probe: trendが整っているが戦略が沈黙のときに小さく試す
            opportunistic_enabled = os.getenv("OPPORTUNISTIC_MACRO", "0").strip().lower() not in {"", "0", "false", "no"}
            if opportunistic_enabled:
                if range_active and not range_macro_bias_dir:
                    logging.info("[OPP] macro probe skipped: range mode without directional bias")
                elif "macro" not in focus_pockets:
                    logging.info(
                        "[OPP] macro probe skipped: macro not in focus pockets (focus=%s pockets=%s)",
                        focus_tag,
                        ",".join(sorted(focus_pockets)),
                    )
                else:
                    has_macro_signal = any(
                        sig for sig in evaluated_signals if sig.get("pocket") == "macro" and sig.get("action") in {"OPEN_LONG", "OPEN_SHORT"}
                    )
                    if has_macro_signal:
                        logging.info("[OPP] macro probe skipped: evaluated signal already present")
                    elif not (fac_h4 and fac_m1):
                        logging.info(
                            "[OPP] macro probe skipped: factors missing (H4=%s M1=%s)",
                            bool(fac_h4),
                            bool(fac_m1),
                        )
                    else:
                        try:
                            ma10_h4 = float(fac_h4.get("ma10") or 0.0)
                            ma20_h4 = float(fac_h4.get("ma20") or 0.0)
                            adx_h4 = float(fac_h4.get("adx") or 0.0)
                            gap_pips_h4 = abs(ma10_h4 - ma20_h4) / PIP
                            rsi_m1_val = float(fac_m1.get("rsi") or 50.0)
                            ema20_m1_val = float(fac_m1.get("ema20") or (fac_m1.get("ma20") or 0.0))
                            close_m1_val = float(fac_m1.get("close") or 0.0)
                        except Exception:
                            ma10_h4 = ma20_h4 = adx_h4 = gap_pips_h4 = rsi_m1_val = ema20_m1_val = close_m1_val = 0.0
                        direction = None
                        overstretched = True
                        if ma10_h4 > ma20_h4:
                            direction = DIRECTION_LONG
                            overstretched = bool(ema20_m1_val and close_m1_val < ema20_m1_val - 0.03)
                        elif ma10_h4 < ma20_h4:
                            direction = DIRECTION_SHORT
                            overstretched = bool(ema20_m1_val and close_m1_val > ema20_m1_val + 0.03)
                        near_trend = (gap_pips_h4 >= 3.0 and adx_h4 >= 15.0)
                        rsi_ok = (35.0 <= rsi_m1_val <= 65.0)
                        if range_active and range_macro_bias_dir:
                            direction = range_macro_bias_dir
                        if direction and near_trend and rsi_ok and not overstretched:
                            atr_hint = fac_m1.get("atr_pips")
                            if atr_hint is None:
                                atr_hint = (fac_m1.get("atr") or 0.0) * 100
                            try:
                                atr_val = float(atr_hint or 0.0)
                            except (TypeError, ValueError):
                                atr_val = 0.0
                            hard_stop = max(18.0, min(36.0, (atr_val or 10.0) * 2.1))
                            tp_soft = max(20.0, min(34.0, hard_stop * 1.05))
                            opp_sig = {
                                "strategy": "OpportunisticMacro",
                                "pocket": "macro",
                                "action": direction,
                                "confidence": 52,
                                # omit sl_pips -> order uses hard_stop_pips; sizing uses capped hard_stop
                                "tp_pips": round(tp_soft, 2),
                                "hard_stop_pips": round(hard_stop, 2),
                                "tag": f"OppMacro-{'bull' if direction=='OPEN_LONG' else 'bear'}",
                                "notes": {
                                    "reason": "opportunistic_trend_probe",
                                    "gap_h4": round(gap_pips_h4, 2),
                                    "adx_h4": round(adx_h4, 2),
                                    "rsi_m1": round(rsi_m1_val, 1),
                                },
                            }
                            if range_active and range_macro_bias_dir:
                                opp_sig["range_bias_dir"] = range_macro_bias_dir
                                opp_sig["notes"]["range_bias_dir"] = range_macro_bias_dir
                                opp_sig["notes"]["range_bias_source"] = "opportunistic_probe"
                            evaluated_signals.append(opp_sig)
                            logging.info(
                                "[OPP] macro probe added dir=%s gap=%.2f adx=%.1f rsi=%.1f",
                                direction,
                                gap_pips_h4,
                                adx_h4,
                                rsi_m1_val,
                            )
                        else:
                            reason = "unknown"
                            detail = ""
                            if direction is None:
                                reason = "trend_alignment"
                                detail = f"ma10={ma10_h4:.4f} ma20={ma20_h4:.4f}"
                            elif not near_trend:
                                reason = "trend_strength"
                                detail = f"gap={gap_pips_h4:.2f} adx={adx_h4:.1f}"
                            elif not rsi_ok:
                                reason = "rsi_window"
                                detail = f"rsi_m1={rsi_m1_val:.1f}"
                            elif overstretched:
                                reason = "overstretched"
                                detail = f"close={close_m1_val:.3f} ema20={ema20_m1_val:.3f}"
                            logging.info(
                                "[OPP] macro probe skipped: %s (%s)",
                                reason,
                                detail,
                            )
            if forced_macro_ctx and not any(sig.get("pocket") == "macro" for sig in evaluated_signals):
                try:
                    fallback = H1MomentumSwing.check(fac_m1)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("[MACRO] forced fallback build failed: %s", exc)
                    fallback = None
                if fallback and fallback.get("action") == forced_macro_ctx.get("direction"):
                    injected = {
                        "strategy": "H1Momentum",
                        "pocket": "macro",
                        "action": fallback.get("action"),
                        "confidence": int(fallback.get("confidence", 60) or 60),
                        "sl_pips": fallback.get("sl_pips"),
                        "tp_pips": fallback.get("tp_pips"),
                        "tag": fallback.get("tag", "H1Momentum"),
                        "hard_stop_pips": fallback.get("hard_stop_pips"),
                        "notes": {
                            "reason": "forced_macro_focus",
                            "source": "H1MomentumFallback",
                            "gap_pips": round(float(forced_macro_ctx.get("gap_pips") or 0.0), 2),
                            "adx_h4": round(float(forced_macro_ctx.get("adx") or 0.0), 1),
                            "rsi_m1": round(float(forced_macro_ctx.get("rsi_m1") or 0.0), 1),
                        },
                    }
                    injected["range_bias_dir"] = fallback.get("action")
                    evaluated_signals.append(injected)
                    logging.info(
                        "[FOCUS] Injected fallback macro signal dir=%s conf=%s sl=%.2f tp=%.2f",
                        injected["action"],
                        injected["confidence"],
                        float(injected.get("sl_pips") or 0.0),
                        float(injected.get("tp_pips") or 0.0),
                    )
                    try:
                        log_metric(
                            "macro_signal_injected",
                            1.0,
                            tags={
                                "source": "forced_macro_focus",
                                "direction": str(forced_macro_ctx.get("direction") or "unknown"),
                            },
                            ts=now,
                        )
                    except Exception:
                        pass
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
                # Only for bot-managed trades; do not touch manual/unknown
                partials = plan_partial_reductions(
                    managed_for_main,
                    fac_m1,
                    fac_h4,
                    range_mode=range_active,
                    now=now,
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
                managed_positions = filter_bot_managed_positions(open_positions)
                managed_for_main = {
                    pocket: info
                    for pocket, info in managed_positions.items()
                    if not _pocket_worker_owns_orders(pocket)
                }
            net_units = int(open_positions.get("__net__", {}).get("units", 0))

            for pocket, info in managed_for_main.items():
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
                        news_cache=news_cache,
                        range_active=range_active,
                        now=now,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("[EXIT_ADVISOR] build_hints failed: %s", exc)
                    advisor_hints = None

            exit_decisions = exit_manager.plan_closures(
                managed_for_main,
                evaluated_signals,
                fac_m1,
                fac_h4,
                event_soon,
                range_active,
                stage_state=stage_snapshot,
                pocket_profiles=recent_profiles,
                now=now,
                stage_tracker=stage_tracker,
            )

            executed_entries: set[tuple[str, str]] = set()
            for decision in exit_decisions:
                pocket = decision.pocket
                remaining = abs(decision.units)
                target_side = "long" if decision.units < 0 else "short"
                trades = (managed_positions.get(pocket, {}) or {}).get("open_trades", [])
                trades = [t for t in trades if t.get("side") == target_side]
                # Clamp requested close units to what's actually available to avoid
                # issuing reduce_only MARKET orders that OANDA will reject with
                # NO_POSITION_TO_REDUCE after we already closed everything.
                try:
                    available = sum(abs(int(t.get("units", 0) or 0)) for t in trades)
                    if available >= 0:
                        remaining = min(remaining, available)
                except Exception:
                    pass
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
                    # Re-check live positions to avoid stale overshoot
                    try:
                        latest_positions = pos_manager.get_open_positions()
                        latest_trades = (
                            (latest_positions.get(pocket, {}) or {}).get("open_trades", [])
                        )
                        latest_trades = [t for t in latest_trades if t.get("side") == target_side]
                        live_available = sum(abs(int(t.get("units", 0) or 0)) for t in latest_trades)
                        if live_available <= 0:
                            remaining = 0
                        else:
                            remaining = min(remaining, live_available)
                    except Exception:
                        pass
                if remaining > 0:
                    client_id = build_client_order_id(focus_tag, decision.tag)
                    fallback_units = -remaining if decision.units < 0 else remaining
                    trade_id, _ = await market_order(
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
                        execute_tag = decision.tag or decision.reason or "exit"
                        executed_entries.add((pocket, execute_tag))
                        cooldown_seconds = cooldown_for_pocket(pocket, range_mode=range_active)
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
            # Use a sizing SL that prefers hard_stop_pips (insurance) but caps macro at SIZING_SL_CAP_MACRO
            sizing_list: list[float] = []
            capped_count = 0
            for s in evaluated_signals:
                raw_sl = s.get("hard_stop_pips")
                if raw_sl is None:
                    raw_sl = s.get("sl_pips")
                if raw_sl is None:
                    continue
                try:
                    val = float(raw_sl)
                except (TypeError, ValueError):
                    continue
                strat_name = s.get("strategy")
                pocket_name = STRATEGIES.get(strat_name).pocket if strat_name in STRATEGIES else s.get("pocket")
                if pocket_name == "macro":
                    capped = min(val, float(SIZING_SL_CAP_MACRO))
                    if capped < val - 1e-9:
                        capped_count += 1
                    val = capped
                sizing_list.append(max(0.5, val))
            avg_sl = sum(sizing_list) / len(sizing_list) if sizing_list else 20.0
            if capped_count:
                logging.info(
                    "[SIZING] capped %d macro SL values at %.2f pips for lot sizing (avg=%.2f)",
                    capped_count,
                    SIZING_SL_CAP_MACRO,
                    avg_sl,
                )

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
            margin_guard_micro = False
            margin_micro_factor = 1.0

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
                if (
                    scalp_buffer is not None
                    and MICRO_MARGIN_GUARD_BUFFER > 0.0
                ):
                    if scalp_buffer < MICRO_MARGIN_GUARD_BUFFER:
                        margin_micro_factor = max(
                            0.25,
                            min(1.0, scalp_buffer / max(MICRO_MARGIN_GUARD_BUFFER, 1e-6)),
                        )
                    if scalp_buffer < MICRO_MARGIN_GUARD_STOP:
                        margin_guard_micro = True
            exposure_state = build_exposure_state(
                open_positions,
                equity=account_equity,
                price=fac_m1.get("close"),
                margin_used=account_snapshot.margin_used if account_snapshot else None,
                margin_available=account_snapshot.margin_available if account_snapshot else None,
                margin_rate=account_snapshot.margin_rate if account_snapshot else None,
            )
            exposure_cap_lot = None
            if exposure_state:
                exposure_cap_lot = max(0.0, exposure_state.available_units() / 100000.0)

            risk_override = _dynamic_risk_pct(
                evaluated_signals,
                range_active,
                weight,
                macro_state,
                now=now,
            )
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
                m1_rsi = None
                m1_rsi_age = None
                try:
                    m1_rsi = float(fac_m1.get("rsi")) if fac_m1 and fac_m1.get("rsi") is not None else None
                except (TypeError, ValueError):
                    m1_rsi = None
                if fac_m1:
                    age_val = fac_m1.get("stale_seconds") or fac_m1.get("stale")
                    try:
                        if age_val is not None:
                            m1_rsi_age = float(age_val)
                    except (TypeError, ValueError):
                        m1_rsi_age = None
                fast_scalp_state.update_from_main(
                    account_equity=account_equity,
                    margin_available=float(margin_available or 0.0),
                    margin_rate=float(margin_rate or 0.0),
                    weight_scalp=weight_scalp,
                    focus_tag=focus_tag,
                    risk_pct_override=risk_override,
                    range_active=range_active,
                    m1_rsi=m1_rsi,
                    m1_rsi_age_sec=m1_rsi_age,
                )

            base_price_val = fac_m1.get("close")
            try:
                mid_price = float(base_price_val or 0.0)
            except (TypeError, ValueError):
                mid_price = 0.0
            if tick_bid is not None and tick_ask is not None:
                mid_price = round((tick_bid + tick_ask) / 2, 3)

            lot_total = allowed_lot(
                account_equity,
                sl_pips=max(1.0, avg_sl),
                margin_available=margin_available,
                price=mid_price if mid_price > 0 else None,
                margin_rate=margin_rate,
                risk_pct_override=risk_override,
            )
            if exposure_cap_lot is not None:
                lot_total = min(lot_total, exposure_cap_lot)
            if (
                exposure_state
                and exposure_cap_lot is not None
                and TARGET_MARGIN_USAGE > 0.0
            ):
                current_usage = max(exposure_state.ratio(), 0.0)
                if current_usage + 1e-6 < TARGET_MARGIN_USAGE:
                    deficit = TARGET_MARGIN_USAGE - current_usage
                    boost = min(
                        MAX_MARGIN_USAGE_BOOST,
                        1.0 + deficit / max(TARGET_MARGIN_USAGE, 1e-3),
                    )
                    boosted = lot_total * boost
                    lot_total = min(boosted, exposure_cap_lot)
                    logging.info(
                        "[EXPOSURE] boosted lot %.3f (ratio %.3f target %.2f boost=%.2f)",
                        lot_total,
                        current_usage,
                        TARGET_MARGIN_USAGE,
                        boost,
                    )
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
            update_dd_context(account_equity, weight, scalp_share)
            lots = alloc(lot_total, weight, scalp_share=scalp_share)
            macro_bias = macro_state.bias(TARGET_INSTRUMENT)
            if (
                MACRO_LOT_BOOST > 1.0
                and lots.get("macro", 0.0) > 0.0
                and weight >= MACRO_LOT_BOOST_WEIGHT
                and abs(macro_bias) >= MACRO_LOT_BIAS_MIN
            ):
                boosted = round(lots["macro"] * MACRO_LOT_BOOST, 3)
                lots["macro"] = boosted
                total_alloc = round(sum(lots.values()), 3)
                target_total = round(lot_total, 3)
                if target_total > 0 and total_alloc > target_total:
                    scale = target_total / total_alloc
                    for key in list(lots.keys()):
                        lots[key] = round(lots[key] * scale, 3)
                logging.info(
                    "[MACRO] boosted pocket bias=%.2f weight=%.2f lot=%.3f",
                    macro_bias,
                    weight,
                    lots["macro"],
                )
            elif lots.get("macro", 0.0) > 0.0:
                bias_strength = abs(macro_bias)
                if bias_strength <= MACRO_LOT_DISABLE_THRESHOLD:
                    logging.info(
                        "[MACRO] bias %.2f neutral; disabling macro pocket for this cycle",
                        macro_bias,
                    )
                    lots["macro"] = 0.0
                elif bias_strength < MACRO_LOT_BIAS_MIN and MACRO_LOT_REDUCTION_FACTOR < 1.0:
                    reduced = round(lots["macro"] * MACRO_LOT_REDUCTION_FACTOR, 3)
                    floor_hint = round(max(0.0, lot_total * MACRO_LOT_MIN_FRACTION), 3)
                    logging.info(
                        "[MACRO] reducing lot due to weak bias %.2f => %.3f (floor %.3f)",
                        macro_bias,
                        reduced,
                        floor_hint,
                    )
                    lots["macro"] = max(floor_hint, reduced)
            if lots.get("micro", 0.0) > 0.0:
                bias_strength = abs(macro_bias)
                if bias_strength <= MICRO_BIAS_DISABLE_THRESHOLD:
                    logging.info(
                        "[MICRO] bias %.2f neutral; suppressing micro entries",
                        macro_bias,
                    )
                    lots["micro"] = 0.0
                elif bias_strength < MICRO_BIAS_REDUCTION_THRESHOLD and MICRO_BIAS_REDUCTION_FACTOR < 1.0:
                    reduced = round(lots["micro"] * MICRO_BIAS_REDUCTION_FACTOR, 3)
                    logging.info(
                        "[MICRO] reducing lot due to weak macro bias %.2f => %.3f",
                        macro_bias,
                        reduced,
                    )
                    lots["micro"] = reduced
            for pocket_key in list(lots.keys()):
                if pocket_key not in focus_pockets:
                    lots[pocket_key] = 0.0
            active_pockets = {sig["pocket"] for sig in evaluated_signals}
            for key in list(lots):
                if key not in active_pockets:
                    lots[key] = 0.0
            if range_active and "macro" in lots:
                lots["macro"] = 0.0
            pocket_plan_notes = {
                "net_units": net_units,
                "spread_live_pips": spread_live_pips,
                "margin_buffer": scalp_buffer,
                "risk_pct": risk_override,
                "macro_snapshot_age": macro_snapshot_age,
                "spread_macro_relaxed": spread_macro_relaxed,
                "spread_micro_relaxed": spread_micro_relaxed,
            }
            _publish_pocket_plans(
                now=now,
                focus_tag=focus_tag,
                focus_pockets=focus_pockets,
                range_active=range_active,
                range_soft_active=range_soft_active,
                range_ctx=range_ctx,
                event_soon=event_soon,
                spread_gate_active=spread_blocked,
                spread_gate_reason=spread_gate_reason,
                spread_log_context=spread_log_context,
                lots=lots,
                risk_override=risk_override,
                weight_macro=weight,
                scalp_share=scalp_share,
                evaluated_signals=evaluated_signals,
                perf_cache=perf_cache,
                fac_m1=fac_m1,
                fac_h4=fac_h4,
                notes=pocket_plan_notes,
                spread_macro_relaxed=spread_macro_relaxed,
                spread_micro_relaxed=spread_micro_relaxed,
            )

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

            spread_skip_logged = False
            usd_long_cap_units = int(_EXPOSURE_USD_LONG_MAX_LOT * 100000)
            projected_usd_long_units = max(0, net_units)
            margin_guard_logged = False
            pocket_allocations: dict[str, dict[str, float]] = defaultdict(lambda: {"long": 0.0, "short": 0.0})
            for signal in evaluated_signals:
                pocket = signal["pocket"]
                action = signal.get("action")
                if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                    continue
                if _pocket_delegated(pocket):
                    continue
                if spread_gate_active:
                    if not spread_skip_logged:
                        logging.info(
                            "[STORY] micro override pocket=%s action=%s macro_trend=%s micro_trend=%s higher_trend=%s",
                            pocket,
                            action,
                            story_snapshot.macro_trend,
                            story_snapshot.micro_trend,
                            story_snapshot.higher_trend,
                        )
                    else:
                        logging.info(
                            "[STORY] skip pocket=%s action=%s trend macro=%s micro=%s",
                            pocket,
                            action,
                            story_snapshot.macro_trend,
                            story_snapshot.micro_trend,
                        )
                        continue
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
                    news_bias_hint=news_bias_hint,
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
                if pocket == "micro" and margin_guard_micro:
                    if not margin_guard_logged:
                        logging.info(
                            "[SKIP] Margin guard active (buffer=%.3f), blocking micro entry.",
                            scalp_buffer if scalp_buffer is not None else -1.0,
                        )
                        margin_guard_logged = True
                    continue
                strategy_name = signal.get("tag") or signal.get("strategy") or "signal"
                if (pocket, strategy_name) in executed_entries:
                    logging.info("[SKIP] %s/%s already handled this loop.", pocket, strategy_name)
                    continue
                macro_bias_dir = signal.get("range_bias_dir")
                if override_macro_hold_active and pocket == "macro":
                    logging.info(
                        "[SKIP] macro_hold_after_breakout, skipping macro entry (score=%.2f momentum=%.4f atr=%.2f vol5m=%.2f override=%s reason=%s hold_until=%s)",
                        range_ctx.score,
                        momentum,
                        atr_pips,
                        vol_5m,
                        override_release_active,
                        range_breakout_reason or range_ctx.reason,
                        (
                            range_macro_hold_until.isoformat(timespec="seconds")
                            if override_macro_hold_active else "n/a"
                        ),
                    )
                    continue
                if range_active and pocket == "macro" and not macro_bias_dir:
                    logging.info(
                        "[SKIP] range_active, no macro bias -> skip macro entry (score=%.2f reason=%s)",
                        range_ctx.score,
                        range_ctx.reason,
                    )
                    continue
                if range_active and pocket == "macro" and macro_bias_dir:
                    logging.info(
                        "[RANGE] macro bias entry allowed dir=%s score=%.2f momentum=%.4f atr=%.2f",
                        macro_bias_dir,
                        range_ctx.score,
                        momentum,
                        atr_pips,
                    )
                if not can_trade(pocket):
                    logging.info(f"[SKIP] DD limit for {pocket} pocket.")
                    continue

                total_lot_for_pocket = lots.get(pocket, 0.0)
                if pocket == "micro":
                    # エントリー優先のため flow/margin スケールを無効化
                    margin_micro_factor = 1.0
                    total_lot_for_pocket = lots.get(pocket, 0.0)
                if total_lot_for_pocket <= 0:
                    continue

                dir_key = "long" if action == "OPEN_LONG" else "short"
                allocated_lot = pocket_allocations[pocket][dir_key]
                remaining_lot = max(0.0, total_lot_for_pocket - allocated_lot)
                if remaining_lot <= 0.0:
                    logging.info(
                        "[POCKET] %s %s allocation exhausted (%.3f lot). Skip %s.",
                        pocket,
                        dir_key,
                        allocated_lot,
                        signal["strategy"],
                    )
                    continue

                confidence = max(0, min(100, signal.get("confidence", 50)))
                base_conf_factor = max(0.3, confidence / 100.0)
                confidence_factor = base_conf_factor
                if pocket == "macro":
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
                if pocket == "scalp":
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
                if confidence_target > remaining_lot:
                    confidence_target = round(remaining_lot, 3)
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
                try:
                    price = float(fac_m1.get("close"))
                except (TypeError, ValueError):
                    logging.debug("[SKIP] Invalid M1 close price: %s", fac_m1.get("close"))
                    continue
                quote_bid = None
                quote_ask = None
                if spread_state:
                    try:
                        quote_bid = float(spread_state.get("bid") or 0.0)
                    except (TypeError, ValueError):
                        quote_bid = None
                    try:
                        quote_ask = float(spread_state.get("ask") or 0.0)
                    except (TypeError, ValueError):
                        quote_ask = None
                if action == "OPEN_LONG":
                    direction = "long"
                    ref_price = open_info.get("long_avg_price") if open_info else None
                else:
                    direction = "short"
                    ref_price = open_info.get("short_avg_price") if open_info else None
                strategy_tag = signal.get("tag") or signal.get("strategy")
                strategy_units, strategy_avg_price = _strategy_position_snapshot(
                    open_info,
                    direction,
                    strategy_tag,
                    price,
                )
                open_units = strategy_units
                if strategy_avg_price is not None:
                    ref_price = strategy_avg_price

                size_factor = stage_tracker.size_multiplier(pocket, direction)
                if size_factor < 0.999:
                    logging.info("[SIZE] %s %s factor=%.2f due to streaks", pocket, direction, size_factor)
                confidence_target = round(confidence_target * size_factor, 3)
                if confidence_target <= 0:
                    continue

                pattern_eval = None
                pattern_tag = signal.get("pattern_tag")
                if pattern_tag:
                    try:
                        pattern_eval = pattern_stats.evaluate(
                            pattern_tag=pattern_tag,
                            pocket=pocket,
                            direction=dir_key,
                            range_mode=range_active,
                            now=now,
                        )
                    except Exception as exc:
                        logging.debug("[PATTERN] evaluate failed tag=%s err=%s", pattern_tag, exc)
                        pattern_eval = None
                if pattern_eval and pattern_eval.factor > 1.0:
                    boosted = round(confidence_target * pattern_eval.factor, 3)
                    boosted = min(boosted, remaining_lot)
                    if boosted > confidence_target:
                        confidence_target = boosted
                        try:
                            log_metric(
                                "pattern_boost",
                                pattern_eval.factor,
                                tags={
                                    "pattern": pattern_tag[:48],
                                    "pocket": pocket,
                                    "dir": dir_key,
                                    "samples": str(pattern_eval.sample_size),
                                    "wr": f"{pattern_eval.win_rate:.3f}",
                                },
                            )
                        except Exception:
                            pass
                        logging.info(
                            "[PATTERN] boost %s pocket=%s dir=%s factor=%.2f wr=%.3f pf=%.3f n=%d",
                            pattern_tag,
                            pocket,
                            dir_key,
                            pattern_eval.factor,
                            pattern_eval.win_rate,
                            pattern_eval.profit_factor,
                            pattern_eval.sample_size,
                        )
                if confidence_target > remaining_lot:
                    confidence_target = round(remaining_lot, 3)
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

                high_impact_context = _evaluate_high_impact_context(
                    pocket=pocket,
                    direction=direction,
                    lose_streak=lose_streak,
                    win_streak=win_streak,
                    fac_m1=fac_m1,
                    fac_h4=fac_h4,
                    story_snapshot=story_snapshot,
                    news_bias_hint=news_bias_hint,
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
                if units == 0:
                    logging.info(
                        "[SKIP] Stage lot %.3f produced 0 units. Skipping.", staged_lot
                    )
                    continue
                if exposure_state and exposure_state.would_exceed(units):
                    logging.info(
                        "[EXPOSURE] cap reached ratio=%.3f limit_lot=%.2f; skip %s",
                        exposure_state.ratio(),
                        exposure_state.limit_units() / 100000.0,
                        signal["strategy"],
                    )
                    continue
                if (
                    usd_long_cap_units > 0
                    and action == "OPEN_LONG"
                    and (projected_usd_long_units + max(0, units)) > usd_long_cap_units
                ):
                    logging.info(
                        "[EXPOSURE] USD long cap %.2f lot reached (projected %.2f). Skip %s.",
                        _EXPOSURE_USD_LONG_MAX_LOT,
                        (projected_usd_long_units + max(0, units)) / 100000.0,
                        signal["strategy"],
                    )
                    continue

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
                    news_bias_hint=news_bias_hint,
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
                    if is_buy:
                        target_price = round(entry_price - pullback_pips * PIP, 3)
                    else:
                        target_price = round(entry_price + pullback_pips * PIP, 3)
                    entry_type = "limit"
                    tolerance_pips = max(
                        0.35,
                        min(
                            1.25,
                            pullback_pips * (0.28 if atr_hint <= 2.0 else 0.33)
                            + (0.18 if abs(momentum) <= 0.008 else 0.08),
                        ),
                    )
                    pullback_note = pullback_pips
                tolerance_price = tolerance_pips * PIP
                reference_price = entry_price
                if entry_type == "limit":
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
                    if is_buy:
                        if price > reference_price + tolerance_price:
                            logging.info(
                                "[LIMIT] Waiting for pullback (target=%.3f cur=%.3f tol=%.2fp)",
                                reference_price,
                                price,
                                tolerance_pips,
                            )
                            continue
                    else:
                        if price < reference_price - tolerance_price:
                            logging.info(
                                "[LIMIT] Waiting for bounce (target=%.3f cur=%.3f tol=%.2fp)",
                                reference_price,
                                price,
                                tolerance_pips,
                            )
                            continue

                sl_pips = signal.get("sl_pips")
                if sl_pips is None:
                    hard_stop = signal.get("hard_stop_pips")
                    if hard_stop is not None:
                        sl_pips = hard_stop
                tp_pips = signal.get("tp_pips")
                if tp_pips is None:
                    logging.info("[SKIP] Missing TP for %s.", signal["strategy"])
                    continue
                if (not DISABLE_STOP_LOSS) and sl_pips is None:
                    logging.info("[SKIP] Missing SL for %s (SL required when DISABLE_STOP_LOSS=false).", signal["strategy"])
                    continue

                entry_price = price
                if action == "OPEN_LONG":
                    if quote_ask:
                        entry_price = quote_ask
                    elif quote_bid:
                        entry_price = quote_bid
                else:
                    if quote_bid:
                        entry_price = quote_bid
                    elif quote_ask:
                        entry_price = quote_ask
                tp_base = entry_price + tp_pips / 100 if action == "OPEN_LONG" else entry_price - tp_pips / 100
                if DISABLE_STOP_LOSS:
                    sl = None
                    tp = round(tp_base, 3)
                else:
                    sl_base = entry_price - sl_pips / 100 if action == "OPEN_LONG" else entry_price + sl_pips / 100
                    sl, tp = clamp_sl_tp(entry_price, sl_base, tp_base, action == "OPEN_LONG")

                    client_id = build_client_order_id(focus_tag, signal["tag"])
                    entry_thesis = {
                        "strategy_tag": signal.get("tag"),
                        "strategy": signal.get("strategy"),
                        "pocket": pocket,
                        "profile": signal.get("profile"),
                        "min_hold_sec": signal.get("min_hold_sec"),
                        "loss_guard_pips": signal.get("loss_guard_pips"),
                        "target_tp_pips": tp_pips,
                        "sl_pips": sl_pips,
                        "tp_pips": tp_pips,
                        "confidence": confidence,
                        "pattern_tag": signal.get("pattern_tag"),
                        "pattern_meta": signal.get("pattern_meta"),
                        "pattern_boost_factor": pattern_eval.factor if pattern_eval else 1.0,
                        "pattern_sample_size": pattern_eval.sample_size if pattern_eval else 0,
                        "pattern_win_rate": pattern_eval.win_rate if pattern_eval else 0.0,
                    }
                    pocket_allocations[pocket][dir_key] += staged_lot
                    trade_id = await market_order(
                        "USD_JPY",
                        units,
                        sl,
                        tp,
                        pocket,
                        client_order_id=client_id,
                        entry_thesis=entry_thesis,
                        meta={"entry_price": entry_price},
                        confidence=confidence,
                    )
                if trade_id:
                    if exposure_state:
                        exposure_state.allocate(units)
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
                    executed_entries.add((pocket, strategy_name))
                    projected_usd_long_units = max(0, net_units)
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

            _publish_policy_snapshot(
                focus_tag=focus,
                focus_pockets=focus_pockets,
                weight_macro=weight,
                macro_regime=macro_regime,
                micro_regime=micro_regime,
                range_ctx=range_ctx,
                range_active=range_active,
                event_soon=event_soon,
                spread_gate_active=spread_gate_active,
                spread_gate_reason=spread_gate_reason,
                spread_macro_relaxed=spread_macro_relaxed,
                spread_micro_relaxed=spread_micro_relaxed,
                lots=lots,
                perf_cache=perf_cache or {},
                managed_positions=managed_positions,
                scalp_share=scalp_share,
                risk_pct=risk_override,
                fac_m1=fac_m1,
                fac_h4=fac_h4,
                strategies_by_pocket={k: list(v) for k, v in strategies_by_pocket.items()},
                micro_hint=micro_gpt_hint,
            )

            # --- 5. 決済済み取引の同期 ---
            recent_trades = pos_manager.sync_trades()
            if recent_trades:
                freeze_candidate, last_exit_info = _apply_trade_cooldowns(
                    recent_trades, stage_tracker, now
                )
                if freeze_candidate and freeze_candidate > trendma_news_cooldown_until:
                    trendma_news_cooldown_until = freeze_candidate
                    logging.info(
                        "[TRENDMA] News freeze extended via recent trades for %.0fs.",
                        (trendma_news_cooldown_until - now).total_seconds(),
                    )
                if last_exit_info:
                    trendma_last_exit = last_exit_info

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
        ("H1", h1_candle_handler),
        ("H4", h4_candle_handler),
    ]
    await initialize_history("USD_JPY")
    try:
        await asyncio.to_thread(
            refresh_macro_snapshot,
            snapshot_path=_macro_snapshot_path(),
            deadzone=_MACRO_STATE_DEADZONE,
            refresh_if_older_than_minutes=int(_MACRO_SNAPSHOT_REFRESH_MINUTES),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("[MACRO] initial snapshot build failed: %s", exc)
    async def _macro_refresher(interval_sec: float = 600.0) -> None:
        while True:
            try:
                await asyncio.to_thread(
                    refresh_macro_snapshot,
                    snapshot_path=_macro_snapshot_path(),
                    deadzone=_MACRO_STATE_DEADZONE,
                    refresh_if_older_than_minutes=int(_MACRO_SNAPSHOT_REFRESH_MINUTES),
                )
                globals()["_macro_state_cache"] = None
                _refresh_macro_state()
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("[MACRO] periodic snapshot refresh failed: %s", exc)
            await asyncio.sleep(max(120.0, interval_sec))

    tasks = [
        start_candle_stream("USD_JPY", handlers),
        logic_loop(),
        news_fetch_loop(),
        summary_ingest_loop(),
        _macro_refresher(),
    ]
    if kaizen_loop is not None:
        logging.info("[KAIZEN] audit loop enabled")
        tasks.append(kaizen_loop())
    else:
        logging.info("[KAIZEN] audit loop unavailable (module not found)")
    scalp_workers = [
        (scalp_core_config.ENABLED, scalp_core_worker, scalp_core_config.LOG_PREFIX),
        (pullback_scalp_config.ENABLED, pullback_scalp_worker, pullback_scalp_config.LOG_PREFIX),
        (pullback_s5_config.ENABLED, pullback_s5_worker, pullback_s5_config.LOG_PREFIX),
        (pullback_runner_s5_config.ENABLED, pullback_runner_s5_worker, pullback_runner_s5_config.LOG_PREFIX),
        (impulse_break_s5_config.ENABLED, impulse_break_s5_worker, impulse_break_s5_config.LOG_PREFIX),
        (impulse_retest_s5_config.ENABLED, impulse_retest_s5_worker, impulse_retest_s5_config.LOG_PREFIX),
        (impulse_momentum_s5_config.ENABLED, impulse_momentum_s5_worker, impulse_momentum_s5_config.LOG_PREFIX),
        (vwap_magnet_s5_config.ENABLED, vwap_magnet_s5_worker, vwap_magnet_s5_config.LOG_PREFIX),
        (squeeze_break_s5_config.ENABLED, squeeze_break_s5_worker, squeeze_break_s5_config.LOG_PREFIX),
        (mirror_spike_config.ENABLED, mirror_spike_worker, mirror_spike_config.LOG_PREFIX),
        (mirror_spike_s5_config.ENABLED, mirror_spike_s5_worker, mirror_spike_s5_config.LOG_PREFIX),
        (mirror_spike_tight_config.ENABLED, mirror_spike_tight_worker, mirror_spike_tight_config.LOG_PREFIX),
        (onepip_maker_s1_config.ENABLED, onepip_maker_s1_worker, onepip_maker_s1_config.LOG_PREFIX),
        (scalp_exit_config.ENABLED, scalp_exit_worker, scalp_exit_config.LOG_PREFIX),
    ]
    micro_workers = [
        (micro_core_config.ENABLED, micro_core_worker, micro_core_config.LOG_PREFIX),
    ]
    macro_workers = [
        (macro_core_config.ENABLED, macro_core_worker, macro_core_config.LOG_PREFIX),
        (trend_h1_config.ENABLED, trend_h1_worker, trend_h1_config.LOG_PREFIX),
        (manual_swing_config.ENABLED, manual_swing_worker, manual_swing_config.LOG_PREFIX),
    ]
    session_workers = [
        (london_momentum_config.ENABLED, london_momentum_worker, london_momentum_config.LOG_PREFIX),
    ]
    scalp_group_enabled = _env_bool("SCALP_WORKERS_ENABLED", True)
    micro_group_enabled = _env_bool("MICRO_WORKERS_ENABLED", True)
    macro_group_enabled = _env_bool("MACRO_WORKERS_ENABLED", True)
    session_group_enabled = _env_bool("SESSION_WORKERS_ENABLED", True)

    if scalp_group_enabled:
        for enabled, worker_fn, prefix in scalp_workers:
            if enabled:
                logging.info("%s bootstrapping worker loop", prefix)
                tasks.append(worker_fn())
            else:
                logging.info("%s worker disabled by configuration", prefix)
    else:
        logging.info("[GROUP] Scalp workers group disabled by configuration")

    if micro_group_enabled:
        for enabled, worker_fn, prefix in micro_workers:
            if enabled:
                logging.info("%s bootstrapping worker loop", prefix)
                tasks.append(worker_fn())
            else:
                logging.info("%s worker disabled by configuration", prefix)
    else:
        logging.info("[GROUP] Micro workers group disabled by configuration")

    if macro_group_enabled:
        for enabled, worker_fn, prefix in macro_workers:
            if enabled:
                logging.info("%s bootstrapping worker loop", prefix)
                tasks.append(worker_fn())
            else:
                logging.info("%s worker disabled by configuration", prefix)
    else:
        logging.info("[GROUP] Macro workers group disabled by configuration")

    if session_group_enabled:
        for enabled, worker_fn, prefix in session_workers:
            if enabled:
                logging.info("%s bootstrapping worker loop", prefix)
                tasks.append(worker_fn())
            else:
                logging.info("%s worker disabled by configuration", prefix)
    else:
        logging.info("[GROUP] Session workers group disabled by configuration")
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
