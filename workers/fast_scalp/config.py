"""
Configuration helpers for the FastScalp worker.
"""

from __future__ import annotations

import os

from execution.stop_loss_policy import stop_loss_disabled

PIP_VALUE = 0.01
STOP_LOSS_DISABLED = stop_loss_disabled()


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


# Hard stop: disable fast scalp worker regardless of environment.
FAST_SCALP_ENABLED: bool = _bool_env("FAST_SCALP_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.1, float(os.getenv("FAST_SCALP_LOOP_INTERVAL_SEC", "0.25")))
TP_BASE_PIPS: float = max(0.2, float(os.getenv("FAST_SCALP_TP_BASE_PIPS", "0.6")))
TP_SPREAD_BUFFER_PIPS: float = max(0.05, float(os.getenv("FAST_SCALP_SPREAD_BUFFER_PIPS", "0.2")))
TP_SAFE_MARGIN_PIPS: float = max(
    0.1, float(os.getenv("FAST_SCALP_TP_SAFE_MARGIN_PIPS", "0.4"))
)
TP_NET_MIN_PIPS: float = max(
    0.2, float(os.getenv("FAST_SCALP_TP_NET_MIN_PIPS", "0.6"))
)
SL_PIPS: float = max(10.0, float(os.getenv("FAST_SCALP_SL_PIPS", "60.0")))
SL_POST_ADJUST_BUFFER_PIPS: float = max(
    0.0, float(os.getenv("FAST_SCALP_SL_POST_ADJUST_BUFFER_PIPS", "5.0"))
)
MAX_SPREAD_PIPS: float = max(0.1, float(os.getenv("FAST_SCALP_MAX_SPREAD_PIPS", "1.3")))
ENTRY_THRESHOLD_PIPS: float = max(0.002, float(os.getenv("FAST_SCALP_ENTRY_MOM_PIPS", "0.003")))
ENTRY_SHORT_THRESHOLD_PIPS: float = max(
    0.002, float(os.getenv("FAST_SCALP_ENTRY_SHORT_MOM_PIPS", "0.003"))
)
ENTRY_RANGE_FLOOR_PIPS: float = max(0.005, float(os.getenv("FAST_SCALP_RANGE_FLOOR_PIPS", "0.02")))
ENTRY_COOLDOWN_SEC: float = max(2.0, float(os.getenv("FAST_SCALP_ENTRY_COOLDOWN_SEC", "3.0")))
MAX_ORDERS_PER_MINUTE: int = max(1, int(float(os.getenv("FAST_SCALP_MAX_ORDERS_PER_MIN", "6"))))
MIN_ORDER_SPACING_SEC: float = max(
    0.5, float(os.getenv("FAST_SCALP_MIN_ORDER_SPACING_SEC", "2.0"))
)
MAX_LOT: float = max(0.001, float(os.getenv("FAST_SCALP_MAX_LOT", "0.1")))
SYNC_INTERVAL_SEC: float = max(5.0, float(os.getenv("FAST_SCALP_SYNC_INTERVAL_SEC", "45.0")))
TIMEOUT_SEC: float = max(10.0, float(os.getenv("FAST_SCALP_TIMEOUT_SEC", "55.0")))
TIMEOUT_MIN_GAIN_PIPS: float = float(os.getenv("FAST_SCALP_TIMEOUT_MIN_GAIN_PIPS", "0.6"))
MAX_DRAWDOWN_CLOSE_PIPS: float = max(
    0.5, float(os.getenv("FAST_SCALP_MAX_DRAWDOWN_CLOSE_PIPS", "1.8"))
)
FAST_SHARE_HINT: float = max(0.0, min(1.0, float(os.getenv("FAST_SCALP_SHARE_HINT", "0.35"))))
SHORT_WINDOW_SEC: float = max(0.2, float(os.getenv("FAST_SCALP_SHORT_WINDOW_SEC", "0.9")))
LONG_WINDOW_SEC: float = max(
    SHORT_WINDOW_SEC + 0.2, float(os.getenv("FAST_SCALP_LONG_WINDOW_SEC", "9.0"))
)
JST_OFF_HOURS_START: int = min(23, max(0, int(float(os.getenv("FAST_SCALP_OFF_HOURS_START_JST", "3")))))
JST_OFF_HOURS_END: int = min(23, max(0, int(float(os.getenv("FAST_SCALP_OFF_HOURS_END_JST", "5")))))
OFF_HOURS_ENABLED: bool = _bool_env("FAST_SCALP_OFF_HOURS_ENABLED", False)
LOG_PREFIX_TICK = "[SCALP-TICK]"
MIN_UNITS: int = max(0, int(float(os.getenv("FAST_SCALP_MIN_UNITS", "1000"))))
MAX_ACTIVE_TRADES: int = max(1, int(float(os.getenv("FAST_SCALP_MAX_ACTIVE", "2"))))
MAX_PER_DIRECTION: int = max(1, int(float(os.getenv("FAST_SCALP_MAX_PER_DIRECTION", "2"))))
STALE_TICK_MAX_SEC: float = max(0.5, float(os.getenv("FAST_SCALP_STALE_TICK_MAX_SEC", "3.0")))
MAX_SIGNAL_AGE_MS: float = max(200.0, float(os.getenv("FAST_SCALP_MAX_SIGNAL_AGE_MS", "6000.0")))
SNAPSHOT_MIN_INTERVAL_SEC: float = max(
    0.25, float(os.getenv("FAST_SCALP_SNAPSHOT_MIN_INTERVAL_SEC", "1.0"))
)
SNAPSHOT_BURST_MAX_ATTEMPTS: int = max(
    1, int(float(os.getenv("FAST_SCALP_SNAPSHOT_BURST_ATTEMPTS", "12")))
)
SNAPSHOT_BURST_INTERVAL_SEC: float = max(
    0.02, float(os.getenv("FAST_SCALP_SNAPSHOT_BURST_INTERVAL_SEC", "0.12"))
)

# 禁止: 損失での自動クローズ（ブローカー側SLは既定で無効）
# True の場合、scalp_fast は含み損の間は worker/exit_manager による決済を行わず、
# 利益確定条件のみでクローズする（グローバルDDや手動クローズは別扱い）。
NO_LOSS_CLOSE: bool = _bool_env("FAST_SCALP_NO_LOSS_CLOSE", True)

# --- entry gating / quality thresholds ---
MIN_ENTRY_ATR_PIPS: float = max(0.0, float(os.getenv("FAST_SCALP_MIN_ENTRY_ATR_PIPS", "0.08")))
MIN_ENTRY_TICK_COUNT: int = max(2, int(float(os.getenv("FAST_SCALP_MIN_ENTRY_TICK_COUNT", "4"))))
RSI_ENTRY_OVERBOUGHT: float = float(os.getenv("FAST_SCALP_RSI_ENTRY_OVERBOUGHT", "70"))
RSI_ENTRY_OVERSOLD: float = float(os.getenv("FAST_SCALP_RSI_ENTRY_OVERSOLD", "30"))
LOW_VOL_COOLDOWN_SEC: float = max(0.0, float(os.getenv("FAST_SCALP_LOW_VOL_COOLDOWN_SEC", "0.0")))
LOW_VOL_MAX_CONSECUTIVE: int = max(1, int(float(os.getenv("FAST_SCALP_LOW_VOL_MAX_CONSECUTIVE", "10"))))
PATTERN_MODEL_PATH: str = os.getenv("FAST_SCALP_PATTERN_MODEL_PATH", "").strip()
PATTERN_MIN_PROB: float = max(0.0, min(1.0, float(os.getenv("FAST_SCALP_PATTERN_MIN_PROB", "0.62"))))
IMPULSE_LOOKBACK_SEC: float = max(
    0.2, float(os.getenv("FAST_SCALP_IMPULSE_WINDOW_SEC", "1.8"))
)
MIN_IMPULSE_PIPS: float = max(
    0.0, float(os.getenv("FAST_SCALP_MIN_IMPULSE_PIPS", "0.7"))
)
IMPULSE_MIN_TICKS: int = max(
    2, int(float(os.getenv("FAST_SCALP_IMPULSE_MIN_TICKS", "6")))
)
CONSOLIDATION_WINDOW_SEC: float = max(
    0.2, float(os.getenv("FAST_SCALP_CONSOLIDATION_WINDOW_SEC", "3.2"))
)
CONSOLIDATION_MAX_RANGE_PIPS: float = max(
    0.0, float(os.getenv("FAST_SCALP_CONSOLIDATION_MAX_RANGE_PIPS", "0.55"))
)
CONSOLIDATION_MIN_TICKS: int = max(
    2, int(float(os.getenv("FAST_SCALP_CONSOLIDATION_MIN_TICKS", "8")))
)
REQUIRE_CONSOLIDATION: bool = _bool_env("FAST_SCALP_REQUIRE_CONSOLIDATION", True)
ENTRY_MIN_VELOCITY_PIPS_PER_SEC: float = max(
    0.0, float(os.getenv("FAST_SCALP_ENTRY_MIN_VELOCITY_PIPS_PER_SEC", "0.0"))
)
ENTRY_MIN_TICK_DENSITY: float = max(
    0.0, float(os.getenv("FAST_SCALP_ENTRY_MIN_TICK_DENSITY", "0.2"))
)
REVERSAL_TO_TREND_IMPULSE_PIPS: float = max(
    0.0, float(os.getenv("FAST_SCALP_REVERSAL_TO_TREND_IMPULSE_PIPS", "1.2"))
)
REVERSAL_TO_TREND_MOM_PIPS: float = max(
    0.0, float(os.getenv("FAST_SCALP_REVERSAL_TO_TREND_MOM_PIPS", "1.0"))
)

# --- technical thresholds ---
RSI_PERIOD: int = max(3, int(float(os.getenv("FAST_SCALP_RSI_PERIOD", "6"))))
RSI_LONG_MIN: float = float(os.getenv("FAST_SCALP_RSI_LONG_MIN", "38"))
RSI_LONG_MAX: float = float(os.getenv("FAST_SCALP_RSI_LONG_MAX", "72"))
RSI_SHORT_MAX: float = float(os.getenv("FAST_SCALP_RSI_SHORT_MAX", "62"))
RSI_SHORT_MIN: float = float(os.getenv("FAST_SCALP_RSI_SHORT_MIN", "32"))
RSI_EXIT_LONG: float = float(os.getenv("FAST_SCALP_RSI_EXIT_LONG", "42"))
RSI_EXIT_SHORT: float = float(os.getenv("FAST_SCALP_RSI_EXIT_SHORT", "58"))

ATR_PERIOD: int = max(2, int(float(os.getenv("FAST_SCALP_ATR_PERIOD", "8"))))
ATR_LOW_VOL_PIPS: float = max(0.0, float(os.getenv("FAST_SCALP_ATR_LOW_VOL_PIPS", "1.4")))
ATR_HIGH_VOL_PIPS: float = max(0.0, float(os.getenv("FAST_SCALP_ATR_HIGH_VOL_PIPS", "4.0")))
# When tick-derived ATR falls below this floor, fall back to an extended window.
ATR_FLOOR_PIPS: float = max(0.0, float(os.getenv("FAST_SCALP_ATR_FLOOR_PIPS", "0.20")))

_MIN_TICK_ENV = int(float(os.getenv("FAST_SCALP_MIN_TICK_COUNT", "14")))
MIN_TICK_COUNT: int = max(RSI_PERIOD + 2, ATR_PERIOD + 1, _MIN_TICK_ENV)

# Require at least ~70% of the sampling window so we avoid gating ourselves out entirely.
_min_span_env = os.getenv("FAST_SCALP_MIN_ENTRY_TICK_SPAN_SEC")
if _min_span_env is None:
    MIN_ENTRY_TICK_SPAN_SEC: float = max(1.0, LONG_WINDOW_SEC * 0.5)
else:
    try:
        MIN_ENTRY_TICK_SPAN_SEC = max(0.0, float(_min_span_env))
    except ValueError:
        MIN_ENTRY_TICK_SPAN_SEC = max(1.0, LONG_WINDOW_SEC * 0.5)
if MIN_ENTRY_TICK_SPAN_SEC >= LONG_WINDOW_SEC:
    MIN_ENTRY_TICK_SPAN_SEC = max(0.0, min(LONG_WINDOW_SEC * 0.95, LONG_WINDOW_SEC - 0.2))

_relax_buffer_env = os.getenv("FAST_SCALP_MIN_SPAN_RELAX_TICK_BUFFER", "6")
try:
    MIN_SPAN_RELAX_TICK_BUFFER: int = max(0, int(float(_relax_buffer_env)))
except ValueError:
    MIN_SPAN_RELAX_TICK_BUFFER = 6
_relax_ratio_env = os.getenv("FAST_SCALP_MIN_SPAN_RELAX_RATIO", "0.55")
try:
    MIN_SPAN_RELAX_RATIO: float = float(_relax_ratio_env)
except ValueError:
    MIN_SPAN_RELAX_RATIO = 0.55
MIN_SPAN_RELAX_RATIO = max(0.1, min(0.95, MIN_SPAN_RELAX_RATIO))

REVIEW_INTERVAL_SEC: float = max(
    0.2, float(os.getenv("FAST_SCALP_REVIEW_INTERVAL_SEC", "1.2"))
)
NEWS_BLOCK_MINUTES: float = float(os.getenv("FAST_SCALP_NEWS_BLOCK_MINUTES", "0"))
MIN_HOLD_SEC: float = max(0.0, float(os.getenv("FAST_SCALP_MIN_HOLD_SEC", "2.5")))
# Watchdog: empty tick streaks before warn/abort (in worker loops)
EMPTY_TICK_WARN_LOOPS: int = max(1, int(float(os.getenv("FAST_SCALP_EMPTY_TICK_WARN_LOOPS", "20"))))
EMPTY_TICK_FATAL_LOOPS: int = max(
    EMPTY_TICK_WARN_LOOPS,
    int(float(os.getenv("FAST_SCALP_EMPTY_TICK_FATAL_LOOPS", "80"))),
)

# Fixed sizing / protections
FIXED_UNITS: int = int(float(os.getenv("FAST_SCALP_FIXED_UNITS", "0")))
USE_SL: bool = False if STOP_LOSS_DISABLED else _bool_env("FAST_SCALP_USE_SL", False)

MAX_MARGIN_USAGE: float = max(0.1, min(1.0, float(os.getenv("FAST_SCALP_MAX_MARGIN_USAGE", "0.9"))))
TIMEOUT_SEC_BASE: float = max(5.0, float(os.getenv("FAST_SCALP_TIMEOUT_SEC_BASE", str(TIMEOUT_SEC))))
TIMEOUT_LOW_VOL_MULT: float = max(0.0, float(os.getenv("FAST_SCALP_TIMEOUT_LOW_VOL_MULT", "0")))
TIMEOUT_HIGH_VOL_MULT: float = max(0.1, float(os.getenv("FAST_SCALP_TIMEOUT_HIGH_VOL_MULT", "0.6")))

M1_RSI_LONG_MIN: float = float(os.getenv("FAST_SCALP_M1_RSI_LONG_MIN", "45"))
M1_RSI_SHORT_MAX: float = float(os.getenv("FAST_SCALP_M1_RSI_SHORT_MAX", "55"))
M1_RSI_CONFIRM_SPAN_SEC: float = max(5.0, float(os.getenv("FAST_SCALP_M1_RSI_CONFIRM_SPAN_SEC", "30.0")))

# --- dynamic entry tuning ---
# エントリー閾値を ATR/スプレッドに応じて可変化
ENTRY_MOM_ATR_COEF: float = max(0.0, float(os.getenv("FAST_SCALP_ENTRY_MOM_ATR_COEF", "0.05")))
ENTRY_MOM_SPREAD_COEF: float = max(0.0, float(os.getenv("FAST_SCALP_ENTRY_MOM_SPREAD_COEF", "0.2")))
ENTRY_RANGE_SPREAD_COEF: float = max(0.0, float(os.getenv("FAST_SCALP_ENTRY_RANGE_SPREAD_COEF", "0.15")))

# テスト/検証用（広いスプレッド時は無効）
FORCE_ENTRIES: bool = _bool_env("FAST_SCALP_FORCE_ENTRIES", True)

# --- advanced timeout control ---
TIMEOUT_EVENT_BUDGET: int = max(1, int(float(os.getenv("FAST_SCALP_TIMEOUT_EVENT_BUDGET", "12"))))
TIMEOUT_EVENT_BUDGET_MAX: int = max(
    TIMEOUT_EVENT_BUDGET, int(float(os.getenv("FAST_SCALP_TIMEOUT_EVENT_BUDGET_MAX", "18")))
)
TIMEOUT_EVENT_HEALTH_EXIT: float = float(
    os.getenv("FAST_SCALP_TIMEOUT_EVENT_HEALTH_EXIT", "0.2")
)
TIMEOUT_HEALTH_KILL_THRESHOLD: float = float(
    os.getenv("FAST_SCALP_TIMEOUT_HEALTH_KILL_THRESHOLD", "0.1")
)
TIMEOUT_HEALTH_EXTEND_THRESHOLD: float = float(
    os.getenv("FAST_SCALP_TIMEOUT_HEALTH_EXTEND_THRESHOLD", "0.5")
)
TIMEOUT_EVENT_EXTEND_EVENTS: int = max(
    0, int(float(os.getenv("FAST_SCALP_TIMEOUT_EVENT_EXTEND_EVENTS", "2")))
)
TIMEOUT_EVENT_EXTEND_SEC: float = max(
    0.0, float(os.getenv("FAST_SCALP_TIMEOUT_EVENT_EXTEND_SEC", "0.3"))
)
TIMEOUT_GRACE_MS: float = max(
    0.0, float(os.getenv("FAST_SCALP_TIMEOUT_GRACE_MS", "600.0"))
)
SCRATCH_MOMENTUM_MIN: float = float(os.getenv("FAST_SCALP_SCRATCH_MOMENTUM_MIN", "0.15"))
SCRATCH_IMBALANCE_MIN: float = float(os.getenv("FAST_SCALP_SCRATCH_IMBALANCE_MIN", "0.12"))
SCRATCH_MAX_SPREAD: float = float(os.getenv("FAST_SCALP_SCRATCH_MAX_SPREAD", "0.40"))
HAZARD_INTERCEPT: float = float(os.getenv("FAST_SCALP_HAZARD_INTERCEPT", "0.25"))
HAZARD_MOMENTUM_COEF: float = float(os.getenv("FAST_SCALP_HAZARD_MOMENTUM_COEF", "1.6"))
HAZARD_IMBALANCE_COEF: float = float(os.getenv("FAST_SCALP_HAZARD_IMBALANCE_COEF", "1.1"))
HAZARD_SPREAD_COEF: float = float(os.getenv("FAST_SCALP_HAZARD_SPREAD_COEF", "1.2"))
HAZARD_LATENCY_COEF: float = float(os.getenv("FAST_SCALP_HAZARD_LATENCY_COEF", "0.0025"))
HAZARD_DEBOUNCE_EVENTS: int = max(
    1, int(float(os.getenv("FAST_SCALP_HAZARD_DEBOUNCE_EVENTS", "4")))
)
HAZARD_COST_SPREAD_BASE: float = max(
    0.05, float(os.getenv("FAST_SCALP_HAZARD_COST_SPREAD_BASE", "0.3"))
)
HAZARD_COST_LATENCY_BASE_MS: float = max(
    50.0, float(os.getenv("FAST_SCALP_HAZARD_COST_LATENCY_BASE_MS", "300.0"))
)
SCRATCH_REQUIRES_EVENTS: int = max(
    0, int(float(os.getenv("FAST_SCALP_SCRATCH_REQUIRES_EVENTS", "0")))
)
TIMEOUT_ADAPTIVE_MIN_SEC: float = max(
    0.5, float(os.getenv("FAST_SCALP_TIMEOUT_ADAPTIVE_MIN_SEC", "1.6"))
)
TIMEOUT_ADAPTIVE_MAX_SEC: float = max(
    TIMEOUT_ADAPTIVE_MIN_SEC, float(os.getenv("FAST_SCALP_TIMEOUT_ADAPTIVE_MAX_SEC", "4.8"))
)
TIMEOUT_FLAT_PIPS_THRESHOLD: float = float(
    os.getenv("FAST_SCALP_TIMEOUT_FLAT_PIPS_THRESHOLD", "0.3")
)
MAX_DRIFT_PIPS: float = max(0.0, float(os.getenv("FAST_SCALP_MAX_DRIFT_PIPS", "0.6")))
DRIFT_TP_RATIO: float = max(0.0, float(os.getenv("FAST_SCALP_DRIFT_TP_RATIO", "0.15")))
TIMEOUT_FLAT_TICKRATE_THRESHOLD: float = float(
    os.getenv("FAST_SCALP_TIMEOUT_FLAT_TICKRATE_THRESHOLD", "6.0")
)
TIMEOUT_ADVERSE_PIPS_THRESHOLD: float = float(
    os.getenv("FAST_SCALP_TIMEOUT_ADVERSE_PIPS_THRESHOLD", "0.5")
)
TIMEOUT_SLIP_PIPS_THRESHOLD: float = float(
    os.getenv("FAST_SCALP_TIMEOUT_SLIP_PIPS_THRESHOLD", "0.25")
)
TIMEOUT_SPREAD_SPIKE_PIPS: float = float(
    os.getenv("FAST_SCALP_TIMEOUT_SPREAD_SPIKE_PIPS", "0.55")
)

# --- quality gates ---
BLOCK_REGIMES = tuple(
    regime.strip()
    for regime in os.getenv("FAST_SCALP_BLOCK_REGIMES", "Event").split(",")
    if regime.strip()
)
LOSS_STREAK_MAX: int = max(
    0, int(float(os.getenv("FAST_SCALP_MAX_CONSEC_LOSSES", "3")))
)
LOSS_STREAK_COOLDOWN_MIN: float = max(
    0.0, float(os.getenv("FAST_SCALP_LOSS_COOLDOWN_MIN", "12"))
)
SESSION_BIAS_ENABLED: bool = _bool_env("FAST_SCALP_SESSION_BIAS_ENABLED", True)
REENTRY_MIN_GAP_SEC: float = max(
    0.0, float(os.getenv("FAST_SCALP_REENTRY_MIN_GAP_SEC", "0.9"))
)
