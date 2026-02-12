from __future__ import annotations

import os

from execution.stop_loss_policy import stop_loss_disabled_for_pocket


ENV_PREFIX = "SCALP_PING_5S"
PIP_VALUE = 0.01


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


ENABLED: bool = _bool_env("SCALP_PING_5S_ENABLED", False)
REQUIRE_PRACTICE: bool = _bool_env("SCALP_PING_5S_REQUIRE_PRACTICE", True)
POCKET: str = os.getenv("SCALP_PING_5S_POCKET", "scalp_fast").strip() or "scalp_fast"
STRATEGY_TAG: str = (
    os.getenv("SCALP_PING_5S_STRATEGY_TAG", "scalp_ping_5s").strip() or "scalp_ping_5s"
)
LOG_PREFIX: str = os.getenv("SCALP_PING_5S_LOG_PREFIX", "[SCALP_PING_5S]")

LOOP_INTERVAL_SEC: float = max(0.05, float(os.getenv("SCALP_PING_5S_LOOP_INTERVAL_SEC", "0.2")))
WINDOW_SEC: float = max(2.0, float(os.getenv("SCALP_PING_5S_WINDOW_SEC", "5.0")))
SIGNAL_WINDOW_SEC: float = max(0.4, float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_SEC", "1.2")))
MIN_TICKS: int = max(4, int(float(os.getenv("SCALP_PING_5S_MIN_TICKS", "12"))))
MIN_SIGNAL_TICKS: int = max(3, int(float(os.getenv("SCALP_PING_5S_MIN_SIGNAL_TICKS", "5"))))
MIN_TICK_RATE: float = max(0.5, float(os.getenv("SCALP_PING_5S_MIN_TICK_RATE", "4.0")))
MAX_TICK_AGE_MS: float = max(100.0, float(os.getenv("SCALP_PING_5S_MAX_TICK_AGE_MS", "800.0")))

MAX_SPREAD_PIPS: float = max(0.1, float(os.getenv("SCALP_PING_5S_MAX_SPREAD_PIPS", "0.35")))
MOMENTUM_TRIGGER_PIPS: float = max(0.1, float(os.getenv("SCALP_PING_5S_MOMENTUM_TRIGGER_PIPS", "0.8")))
MOMENTUM_SPREAD_MULT: float = max(0.0, float(os.getenv("SCALP_PING_5S_MOMENTUM_SPREAD_MULT", "1.0")))
IMBALANCE_MIN: float = max(0.5, min(0.95, float(os.getenv("SCALP_PING_5S_IMBALANCE_MIN", "0.60"))))

DIRECTION_BIAS_ENABLED: bool = _bool_env("SCALP_PING_5S_DIRECTION_BIAS_ENABLED", True)
DIRECTION_BIAS_WINDOW_SEC: float = max(
    SIGNAL_WINDOW_SEC,
    float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_WINDOW_SEC", "20.0")),
)
DIRECTION_BIAS_MIN_TICKS: int = max(
    MIN_SIGNAL_TICKS,
    int(float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_MIN_TICKS", "24"))),
)
DIRECTION_BIAS_NEUTRAL_SCORE: float = max(
    0.05,
    min(0.9, float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_NEUTRAL_SCORE", "0.18"))),
)
DIRECTION_BIAS_BLOCK_SCORE: float = max(
    DIRECTION_BIAS_NEUTRAL_SCORE,
    min(0.98, float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_BLOCK_SCORE", "0.58"))),
)
DIRECTION_BIAS_ALIGN_SCORE_MIN: float = max(
    DIRECTION_BIAS_NEUTRAL_SCORE,
    min(0.95, float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_ALIGN_SCORE_MIN", "0.40"))),
)
DIRECTION_BIAS_OPPOSITE_UNITS_MULT: float = max(
    0.1,
    min(1.0, float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_OPPOSITE_UNITS_MULT", "0.72"))),
)
DIRECTION_BIAS_ALIGN_UNITS_BOOST_MAX: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_ALIGN_UNITS_BOOST_MAX", "0.20"))),
)
DIRECTION_BIAS_VOL_LOW_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_VOL_LOW_PIPS", "0.40"))
)
DIRECTION_BIAS_VOL_HIGH_PIPS: float = max(
    DIRECTION_BIAS_VOL_LOW_PIPS + 0.05,
    float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_VOL_HIGH_PIPS", "2.40")),
)
DIRECTION_BIAS_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_DIRECTION_BIAS_LOG_INTERVAL_SEC", "8.0")),
)

ENTRY_COOLDOWN_SEC: float = max(0.1, float(os.getenv("SCALP_PING_5S_ENTRY_COOLDOWN_SEC", "2.0")))
MIN_ORDER_SPACING_SEC: float = max(
    0.05, float(os.getenv("SCALP_PING_5S_MIN_ORDER_SPACING_SEC", "1.0"))
)
MAX_ORDERS_PER_MINUTE: int = max(
    1, int(float(os.getenv("SCALP_PING_5S_MAX_ORDERS_PER_MINUTE", "30")))
)
MAX_ACTIVE_TRADES: int = max(
    1, int(float(os.getenv("SCALP_PING_5S_MAX_ACTIVE_TRADES", "2")))
)
MAX_PER_DIRECTION: int = max(
    1, int(float(os.getenv("SCALP_PING_5S_MAX_PER_DIRECTION", "1")))
)
NO_HEDGE_ENTRY: bool = _bool_env("SCALP_PING_5S_NO_HEDGE_ENTRY", True)

TRAP_BYPASS_NO_HEDGE: bool = _bool_env("SCALP_PING_5S_TRAP_BYPASS_NO_HEDGE", True)
TRAP_REQUIRE_NET_LOSS: bool = _bool_env("SCALP_PING_5S_TRAP_REQUIRE_NET_LOSS", True)
TRAP_MIN_LONG_UNITS: int = max(
    0, int(float(os.getenv("SCALP_PING_5S_TRAP_MIN_LONG_UNITS", "8000")))
)
TRAP_MIN_SHORT_UNITS: int = max(
    0, int(float(os.getenv("SCALP_PING_5S_TRAP_MIN_SHORT_UNITS", "8000")))
)
TRAP_MAX_NET_RATIO: float = max(
    0.0, min(1.0, float(os.getenv("SCALP_PING_5S_TRAP_MAX_NET_RATIO", "0.45")))
)
TRAP_MIN_COMBINED_DD_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_TRAP_MIN_COMBINED_DD_PIPS", "0.8"))
)
TRAP_LOG_INTERVAL_SEC: float = max(
    1.0, float(os.getenv("SCALP_PING_5S_TRAP_LOG_INTERVAL_SEC", "20.0"))
)

MIN_UNITS: int = max(100, int(float(os.getenv("SCALP_PING_5S_MIN_UNITS", "2000"))))
MAX_UNITS: int = max(MIN_UNITS, int(float(os.getenv("SCALP_PING_5S_MAX_UNITS", "25000"))))
BASE_ENTRY_UNITS: int = max(
    MIN_UNITS, int(float(os.getenv("SCALP_PING_5S_BASE_ENTRY_UNITS", "10000")))
)
MIN_FREE_MARGIN_RATIO: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_MIN_FREE_MARGIN_RATIO", "0.03"))
)

CONFIDENCE_FLOOR: int = max(0, int(float(os.getenv("SCALP_PING_5S_CONF_FLOOR", "58"))))
CONFIDENCE_CEIL: int = max(
    CONFIDENCE_FLOOR + 1, int(float(os.getenv("SCALP_PING_5S_CONF_CEIL", "92")))
)

TP_BASE_PIPS: float = max(0.2, float(os.getenv("SCALP_PING_5S_TP_BASE_PIPS", "0.2")))
TP_NET_MIN_PIPS: float = max(0.1, float(os.getenv("SCALP_PING_5S_TP_NET_MIN_PIPS", "0.25")))
TP_MAX_PIPS: float = max(TP_BASE_PIPS, float(os.getenv("SCALP_PING_5S_TP_MAX_PIPS", "1.0")))
TP_MOMENTUM_BONUS_MAX: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_TP_MOMENTUM_BONUS_MAX", "0.2"))
)

SL_BASE_PIPS: float = max(0.2, float(os.getenv("SCALP_PING_5S_SL_BASE_PIPS", "2.4")))
SL_MIN_PIPS: float = max(0.2, float(os.getenv("SCALP_PING_5S_SL_MIN_PIPS", "1.9")))
SL_MAX_PIPS: float = max(SL_MIN_PIPS, float(os.getenv("SCALP_PING_5S_SL_MAX_PIPS", "6.0")))
SL_SPREAD_MULT: float = max(0.0, float(os.getenv("SCALP_PING_5S_SL_SPREAD_MULT", "1.8")))
SL_SPREAD_BUFFER_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_SL_SPREAD_BUFFER_PIPS", "0.4"))
)

SNAPSHOT_FALLBACK_ENABLED: bool = _bool_env("SCALP_PING_5S_SNAPSHOT_FALLBACK_ENABLED", True)
SNAPSHOT_MIN_INTERVAL_SEC: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_SNAPSHOT_MIN_INTERVAL_SEC", "0.2"))
)

FORCE_EXIT_ENABLED: bool = _bool_env("SCALP_PING_5S_FORCE_EXIT_ENABLED", True)
FORCE_EXIT_MAX_ACTIONS: int = max(
    0, int(float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_ACTIONS", "0")))
)
FORCE_EXIT_MAX_HOLD_SEC: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_HOLD_SEC", "0"))
)
FORCE_EXIT_REASON_TIME: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_REASON", "time_stop").strip() or "time_stop"
)
FORCE_EXIT_MAX_FLOATING_LOSS_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS", "0"))
)
FORCE_EXIT_REASON_MAX_FLOATING_LOSS: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_FLOATING_LOSS_REASON", "max_floating_loss").strip()
    or "max_floating_loss"
)
FORCE_EXIT_RECOVERY_WINDOW_SEC: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_RECOVERY_WINDOW_SEC", "0"))
)
FORCE_EXIT_RECOVERABLE_LOSS_PIPS: float = float(
    os.getenv("SCALP_PING_5S_FORCE_EXIT_RECOVERABLE_LOSS_PIPS", "0")
)
FORCE_EXIT_REASON_RECOVERY: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_RECOVERY_REASON", "no_recovery").strip()
    or "no_recovery"
)
FORCE_EXIT_GIVEBACK_ENABLED: bool = _bool_env("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_ENABLED", False)
FORCE_EXIT_GIVEBACK_ARM_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_ARM_PIPS", "0"))
)
FORCE_EXIT_GIVEBACK_BACKOFF_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_BACKOFF_PIPS", "0"))
)
FORCE_EXIT_GIVEBACK_MIN_HOLD_SEC: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_MIN_HOLD_SEC", "0"))
)
FORCE_EXIT_GIVEBACK_PROTECT_PIPS: float = float(
    os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_PROTECT_PIPS", "0")
)
FORCE_EXIT_REASON_GIVEBACK: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_REASON", "giveback_lock").strip()
    or "giveback_lock"
)
FORCE_EXIT_REQUIRE_POLICY_GENERATION: bool = _bool_env(
    "SCALP_PING_5S_FORCE_EXIT_REQUIRE_POLICY_GENERATION",
    False,
)
FORCE_EXIT_POLICY_GENERATION: str = os.getenv(
    "SCALP_PING_5S_FORCE_EXIT_POLICY_GENERATION",
    "",
).strip()
# Keep current live positions untouched after worker restart; force-exit applies only to new trades.
FORCE_EXIT_SKIP_EXISTING_ON_START: bool = _bool_env(
    "SCALP_PING_5S_FORCE_EXIT_SKIP_EXISTING_ON_START",
    True,
)
FORCE_EXIT_ACTIVE: bool = FORCE_EXIT_ENABLED and FORCE_EXIT_MAX_ACTIONS > 0 and (
    FORCE_EXIT_MAX_HOLD_SEC > 0.0
    or FORCE_EXIT_MAX_FLOATING_LOSS_PIPS > 0.0
    or FORCE_EXIT_RECOVERY_WINDOW_SEC > 0.0
    or (
        FORCE_EXIT_GIVEBACK_ENABLED
        and FORCE_EXIT_GIVEBACK_ARM_PIPS > 0.0
        and FORCE_EXIT_GIVEBACK_BACKOFF_PIPS > 0.0
    )
)

STOP_LOSS_DISABLED = stop_loss_disabled_for_pocket(POCKET)
USE_SL: bool = False if STOP_LOSS_DISABLED else _bool_env("SCALP_PING_5S_USE_SL", True)
