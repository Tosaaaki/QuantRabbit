import asyncio
import datetime
import logging
import os
import sqlite3
import subprocess
import traceback
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from collections import defaultdict

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
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.news.spike_reversal import NewsSpikeReversal
from strategies.scalping.m1_scalper import M1Scalper
from strategies.scalping.range_fader import RangeFader
from strategies.scalping.pulse_break import PulseBreak
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

FALLBACK_EQUITY = 10000.0  # REST失敗時のフォールバック

RANGE_MACRO_WEIGHT_CAP = 0.22
RANGE_CONFIDENCE_SCALE = {
    "macro": 0.65,
    "micro": 0.85,
    "scalp": 0.75,
}
RANGE_SCALP_ATR_MIN = 1.8
RANGE_SCALP_MOMENTUM_MIN = 0.001
RANGE_SCALP_VOL_MIN = 0.9
SOFT_RANGE_SUPPRESS_STRATEGIES = {"TrendMA", "Donchian55"}
LOW_TREND_ADX_THRESHOLD = 18.0
LOW_TREND_SLOPE_THRESHOLD = 0.00035
LOW_TREND_WEIGHT_CAP = 0.35
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
MANUAL_SENTINEL_POCKETS = {"manual", "unknown"}
MANUAL_SENTINEL_MIN_UNITS = int(os.getenv("MANUAL_SENTINEL_MIN_UNITS", "4000"))
MANUAL_SENTINEL_BLOCK_POCKETS = {"micro", "scalp"}
MANUAL_SENTINEL_RELEASE_CYCLES = max(
    1, int(os.getenv("MANUAL_SENTINEL_RELEASE_CYCLES", "2"))
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
    base = DEFAULT_MIN_HOLD_SEC.get(pocket, 90.0)
    tp = _to_float(signal.get("tp_pips")) or 0.0
    atr_hint = _atr_hint_pips(fac_m1)
    # Scale baseline by TP demand and prevailing ATR so larger swings get more time.
    scaled = base
    if tp > 0:
        scaled = max(scaled, min(600.0, tp * 45.0))
    if atr_hint > 0:
        scaled = max(scaled, min(600.0, atr_hint * 15.0))
    return round(scaled, 1)


def _derive_loss_guard_pips(signal: Dict[str, Any], pocket: str) -> float:
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
    return round(max(0.3, guard), 2)


def _extract_profile_name(raw_signal: Dict[str, Any], strategy_name: str) -> str:
    profile = raw_signal.get("profile")
    if isinstance(profile, str) and profile.strip():
        return profile.strip()
    return strategy_name


def _manual_sentinel_state(open_positions: Dict[str, Dict]) -> tuple[bool, int, str]:
    total_units = 0
    pockets: list[str] = []
    for name in MANUAL_SENTINEL_POCKETS:
        info = open_positions.get(name) or {}
        units = int(abs(info.get("units", 0) or 0))
        if units > 0:
            total_units += units
            pockets.append(f"{name}:{units}")
    active = total_units >= MANUAL_SENTINEL_MIN_UNITS
    details = ",".join(pockets)
    return active, total_units, details
SOFT_RANGE_WEIGHT_CAP = 0.32
SOFT_RANGE_ADX_BUFFER = 6.0
RANGE_ENTRY_CONFIRMATIONS = 2
RANGE_EXIT_CONFIRMATIONS = 3
RANGE_MIN_ACTIVE_SECONDS = 240
RANGE_ENTRY_SCORE_FLOOR = 0.62
RANGE_EXIT_SCORE_CEIL = 0.56
STAGE_RESET_GRACE_SECONDS = 180
TARGET_INSTRUMENT = "USD_JPY"

# 総合的な Macro 配分上限（lot 按分に直接適用）
# 初期値は 0.30 (=30%)。環境変数による上書きは下段で適用。
GLOBAL_MACRO_WEIGHT_CAP = 0.30


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
_MACRO_STATE_STALE_WARN_SEC = _env_float("MACRO_STATE_STALE_WARN_SEC", 900.0)
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
        "python3",
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
            and not spread_gate_active
        )
        if pocket in {"micro", "scalp"} and event_soon:
            entry_allow = False
        entry_gates = {
            "allow_new": entry_allow,
            "require_retest": range_active and pocket != "macro",
            "spread_ok": not spread_gate_active,
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
        )
        plan_bus.publish(plan)


_ORDERS_DB_PATH = Path("logs/orders.db")


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
    perf_cache = {}
    news_cache = {}
    insight = InsightClient()
    missing_factor_cycles = 0
    last_update_time = datetime.datetime.min
    last_heartbeat_time = datetime.datetime.min  # Add this line
    last_metrics_refresh = datetime.datetime.min
    last_macro_snapshot_refresh = datetime.datetime.min
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
    market_closed_logged = False
    last_market_closed_log = datetime.datetime.min
    trendma_news_cooldown_until = datetime.datetime.min
    trendma_last_exit: Optional[Dict[str, object]] = None
    last_hold_ratio_check = datetime.datetime.min

    try:
        while True:
            now = datetime.datetime.utcnow()
            stage_tracker.clear_expired(now)
            stage_tracker.update_loss_streaks(now=now, cooldown_map=POCKET_LOSS_COOLDOWNS)

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
            if macro_snapshot_stale and "macro" in focus_pockets:
                focus_pockets.discard("macro")
                logging.info(
                    "[MACRO] Disabled macro pocket until snapshot refresh (age=%.1fs).",
                    macro_snapshot_age,
                )
            if not focus_pockets:
                focus_pockets = {"micro"}

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
                and "scalp" in focus_pockets
                and "M1Scalper" not in ranked_strategies
            ):
                ranked_strategies.append("M1Scalper")
                logging.info(
                    "[SCALP] Auto-added M1Scalper (range=%s ATR %.2f, momentum %.4f, vol5m %.2f).",
                    range_active,
                    atr_pips,
                    momentum,
                    fac_m1.get("vol_5m", 0.0),
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
                signal["min_hold_sec"] = _derive_min_hold_seconds(
                    signal, cls.pocket, fac_m1
                )
                signal["loss_guard_pips"] = _derive_loss_guard_pips(signal, cls.pocket)
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
            # Use filtered positions for bot-controlled actions
            managed_positions = filter_bot_managed_positions(open_positions)
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
                managed_for_main,
                evaluated_signals,
                fac_m1,
                fac_h4,
                event_soon,
                range_active,
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

            lot_total = allowed_lot(
                account_equity,
                sl_pips=max(1.0, avg_sl),
                margin_available=margin_available,
                price=fac_m1.get("close"),
                margin_rate=margin_rate,
                risk_pct_override=risk_override,
            )
            if exposure_cap_lot is not None:
                lot_total = min(lot_total, exposure_cap_lot)
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
            pocket_plan_notes = {
                "net_units": net_units,
                "spread_live_pips": spread_live_pips,
                "margin_buffer": scalp_buffer,
                "risk_pct": risk_override,
                "macro_snapshot_age": macro_snapshot_age,
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
            )

            spread_skip_logged = False
            usd_long_cap_units = int(_EXPOSURE_USD_LONG_MAX_LOT * 100000)
            projected_usd_long_units = max(0, net_units)
            margin_guard_logged = False
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
                            "[SKIP] Spread guard active (%s, %s).",
                            spread_gate_reason,
                            spread_log_context,
                        )
                        spread_skip_logged = True
                    continue
                if event_soon and pocket in {"micro", "scalp"}:
                    logging.info("[SKIP] Event soon, skipping %s pocket trade.", pocket)
                    continue
                if margin_guard_micro and pocket == "micro":
                    if not margin_guard_logged:
                        logging.warning(
                            "[MARGIN] Micro guard active buffer=%.3f (stop=%.3f)",
                            scalp_buffer if scalp_buffer is not None else -1.0,
                            MICRO_MARGIN_GUARD_STOP,
                        )
                        margin_guard_logged = True
                    continue
                strategy_name = signal.get("tag") or signal.get("strategy") or "signal"
                if (pocket, strategy_name) in executed_entries:
                    logging.info("[SKIP] %s/%s already handled this loop.", pocket, strategy_name)
                    continue
                if range_active and pocket == "macro":
                    logging.info("[SKIP] Range mode active, skipping macro entry.")
                    continue
                if not can_trade(pocket):
                    logging.info(f"[SKIP] DD limit for {pocket} pocket.")
                    continue

                total_lot_for_pocket = lots.get(pocket, 0.0)
                if pocket == "micro":
                    flow_factor = _micro_flow_factor(spread_live_pips, scalp_buffer)
                    if flow_factor < 0.999:
                        adjusted_lot = round(total_lot_for_pocket * flow_factor, 4)
                        reasons: list[str] = []
                        if spread_live_pips is not None:
                            reasons.append(f"spread={spread_live_pips:.2f}p")
                        if (
                            scalp_buffer is not None
                            and scalp_buffer < MICRO_MARGIN_BUFFER_LIMIT
                        ):
                            reasons.append(f"buffer={scalp_buffer:.3f}")
                        if adjusted_lot <= 0:
                            logging.info(
                                "[MICRO] Lot scaled to zero (factor=%.2f, %s); skip entry.",
                                flow_factor,
                                ", ".join(reasons) or "pressure",
                            )
                            continue
                        logging.info(
                            "[MICRO] Lot scaled %.4f -> %.4f (factor=%.2f, %s)",
                            total_lot_for_pocket,
                            adjusted_lot,
                            flow_factor,
                            ", ".join(reasons) or "pressure",
                        )
                        total_lot_for_pocket = adjusted_lot
                    if margin_micro_factor < 0.999 and total_lot_for_pocket > 0:
                        adjusted_lot = round(total_lot_for_pocket * margin_micro_factor, 4)
                        logging.info(
                            "[MICRO] Lot margin scaling %.4f -> %.4f (buffer=%.3f)",
                            total_lot_for_pocket,
                            adjusted_lot,
                            scalp_buffer if scalp_buffer is not None else -1.0,
                        )
                        total_lot_for_pocket = adjusted_lot
                    if margin_guard_micro:
                        logging.info(
                            "[MICRO] Lot forced to zero due to margin guard (buffer=%.3f)",
                            scalp_buffer if scalp_buffer is not None else -1.0,
                        )
                        continue
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

                sl_pips = signal.get("sl_pips")
                tp_pips = signal.get("tp_pips")
                if sl_pips is None or tp_pips is None:
                    logging.info("[SKIP] Missing SL/TP for %s.", signal["strategy"])
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
                sl_base = entry_price - sl_pips / 100 if action == "OPEN_LONG" else entry_price + sl_pips / 100
                tp_base = entry_price + tp_pips / 100 if action == "OPEN_LONG" else entry_price - tp_pips / 100
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
                }
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl,
                    tp,
                    pocket,
                    client_order_id=client_id,
                    entry_thesis=entry_thesis,
                    meta={"entry_price": entry_price},
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
