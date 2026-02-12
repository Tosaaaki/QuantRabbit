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


def _csv_env(name: str) -> tuple[str, ...]:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return ()
    tokens: list[str] = []
    for part in raw.replace("\n", ",").split(","):
        token = str(part).strip()
        if token:
            tokens.append(token)
    return tuple(dict.fromkeys(tokens))


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

MAX_SPREAD_PIPS: float = max(0.1, float(os.getenv("SCALP_PING_5S_MAX_SPREAD_PIPS", "0.85")))
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

LOOKAHEAD_GATE_ENABLED: bool = _bool_env("SCALP_PING_5S_LOOKAHEAD_GATE_ENABLED", True)
LOOKAHEAD_ALLOW_THIN_EDGE: bool = _bool_env("SCALP_PING_5S_LOOKAHEAD_ALLOW_THIN_EDGE", True)
LOOKAHEAD_HORIZON_SEC: float = max(
    0.5,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_HORIZON_SEC", "3.0")),
)
LOOKAHEAD_EDGE_MIN_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_EDGE_MIN_PIPS", "0.10")),
)
LOOKAHEAD_EDGE_REF_PIPS: float = max(
    0.05,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_EDGE_REF_PIPS", "0.60")),
)
LOOKAHEAD_UNITS_MIN_MULT: float = max(
    0.1,
    min(1.0, float(os.getenv("SCALP_PING_5S_LOOKAHEAD_UNITS_MIN_MULT", "0.55"))),
)
LOOKAHEAD_UNITS_MAX_MULT: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_UNITS_MAX_MULT", "1.25")),
)
LOOKAHEAD_MOMENTUM_DECAY: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_MOMENTUM_DECAY", "0.60")),
)
LOOKAHEAD_MOMENTUM_WEIGHT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_MOMENTUM_WEIGHT", "0.72")),
)
LOOKAHEAD_FLOW_WEIGHT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_FLOW_WEIGHT", "0.30")),
)
LOOKAHEAD_RATE_WEIGHT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_RATE_WEIGHT", "0.22")),
)
LOOKAHEAD_BIAS_WEIGHT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_BIAS_WEIGHT", "0.32")),
)
LOOKAHEAD_TRIGGER_WEIGHT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_TRIGGER_WEIGHT", "0.25")),
)
LOOKAHEAD_COUNTER_PENALTY: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_COUNTER_PENALTY", "0.80")),
)
LOOKAHEAD_SLIP_BASE_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_SLIP_BASE_PIPS", "0.05")),
)
LOOKAHEAD_SLIP_SPREAD_MULT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_SLIP_SPREAD_MULT", "0.28")),
)
LOOKAHEAD_SLIP_RANGE_MULT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_SLIP_RANGE_MULT", "0.14")),
)
LOOKAHEAD_LATENCY_PENALTY_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_LATENCY_PENALTY_PIPS", "0.03")),
)
LOOKAHEAD_SAFETY_MARGIN_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_SAFETY_MARGIN_PIPS", "0.02")),
)
LOOKAHEAD_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_LOOKAHEAD_LOG_INTERVAL_SEC", "8.0")),
)

REVERT_ENABLED: bool = _bool_env("SCALP_PING_5S_REVERT_ENABLED", True)
REVERT_WINDOW_SEC: float = max(
    SIGNAL_WINDOW_SEC,
    float(os.getenv("SCALP_PING_5S_REVERT_WINDOW_SEC", "2.8")),
)
REVERT_SHORT_WINDOW_SEC: float = max(
    0.2,
    min(
        REVERT_WINDOW_SEC,
        float(os.getenv("SCALP_PING_5S_REVERT_SHORT_WINDOW_SEC", "0.8")),
    ),
)
REVERT_MIN_TICKS: int = max(
    MIN_SIGNAL_TICKS,
    int(float(os.getenv("SCALP_PING_5S_REVERT_MIN_TICKS", "8"))),
)
REVERT_MIN_TICK_RATE: float = max(
    0.5,
    float(os.getenv("SCALP_PING_5S_REVERT_MIN_TICK_RATE", "4.0")),
)
REVERT_RANGE_MIN_PIPS: float = max(
    0.1,
    float(os.getenv("SCALP_PING_5S_REVERT_RANGE_MIN_PIPS", "0.9")),
)
REVERT_SWEEP_MIN_PIPS: float = max(
    0.1,
    float(os.getenv("SCALP_PING_5S_REVERT_SWEEP_MIN_PIPS", "0.55")),
)
REVERT_BOUNCE_MIN_PIPS: float = max(
    0.05,
    float(os.getenv("SCALP_PING_5S_REVERT_BOUNCE_MIN_PIPS", "0.20")),
)
REVERT_CONFIRM_TICKS: int = max(
    2,
    int(float(os.getenv("SCALP_PING_5S_REVERT_CONFIRM_TICKS", "6"))),
)
REVERT_CONFIRM_RATIO_MIN: float = max(
    0.5,
    min(0.95, float(os.getenv("SCALP_PING_5S_REVERT_CONFIRM_RATIO_MIN", "0.60"))),
)
MODE_SWITCH_REVERT_DOMINANCE: float = max(
    0.8,
    float(os.getenv("SCALP_PING_5S_MODE_SWITCH_REVERT_DOMINANCE", "1.08")),
)
REVERT_DIRECTION_HARD_BLOCK_SCORE: float = max(
    DIRECTION_BIAS_BLOCK_SCORE,
    min(
        0.99,
        float(
            os.getenv(
                "SCALP_PING_5S_REVERT_DIRECTION_HARD_BLOCK_SCORE",
                "0.82",
            )
        ),
    ),
)
REVERT_DIRECTION_OPPOSITE_UNITS_MULT: float = max(
    DIRECTION_BIAS_OPPOSITE_UNITS_MULT,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_REVERT_DIRECTION_OPPOSITE_UNITS_MULT",
                "0.86",
            )
        ),
    ),
)
REVERT_SIDE_BIAS_PENALTY_WEIGHT: float = max(
    0.0,
    min(
        1.0,
        float(os.getenv("SCALP_PING_5S_REVERT_SIDE_BIAS_PENALTY_WEIGHT", "0.45")),
    ),
)
REVERT_SIDE_BIAS_FLOOR: float = max(
    0.1,
    min(
        1.0,
        float(os.getenv("SCALP_PING_5S_REVERT_SIDE_BIAS_FLOOR", "0.55")),
    ),
)

MTF_REGIME_ENABLED: bool = _bool_env("SCALP_PING_5S_MTF_REGIME_ENABLED", True)
MTF_TREND_NEUTRAL_SCORE: float = max(
    0.05,
    min(0.6, float(os.getenv("SCALP_PING_5S_MTF_TREND_NEUTRAL_SCORE", "0.14"))),
)
MTF_TREND_STRONG_SCORE: float = max(
    MTF_TREND_NEUTRAL_SCORE,
    min(0.95, float(os.getenv("SCALP_PING_5S_MTF_TREND_STRONG_SCORE", "0.28"))),
)
MTF_HEAT_CONTINUATION_MIN: float = max(
    0.05,
    min(0.98, float(os.getenv("SCALP_PING_5S_MTF_HEAT_CONTINUATION_MIN", "0.62"))),
)
MTF_HEAT_REVERSION_MAX: float = max(
    0.0,
    min(MTF_HEAT_CONTINUATION_MIN, float(os.getenv("SCALP_PING_5S_MTF_HEAT_REVERSION_MAX", "0.40"))),
)
MTF_ADX_LOW: float = max(1.0, float(os.getenv("SCALP_PING_5S_MTF_ADX_LOW", "16.0")))
MTF_ADX_HIGH: float = max(
    MTF_ADX_LOW + 0.1,
    float(os.getenv("SCALP_PING_5S_MTF_ADX_HIGH", "34.0")),
)
MTF_ATR_M1_LOW_PIPS: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_MTF_ATR_M1_LOW_PIPS", "3.0"))
)
MTF_ATR_M1_HIGH_PIPS: float = max(
    MTF_ATR_M1_LOW_PIPS + 0.1,
    float(os.getenv("SCALP_PING_5S_MTF_ATR_M1_HIGH_PIPS", "9.0")),
)
MTF_ATR_M5_LOW_PIPS: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_MTF_ATR_M5_LOW_PIPS", "6.0"))
)
MTF_ATR_M5_HIGH_PIPS: float = max(
    MTF_ATR_M5_LOW_PIPS + 0.1,
    float(os.getenv("SCALP_PING_5S_MTF_ATR_M5_HIGH_PIPS", "18.0")),
)
MTF_CONTINUATION_ALIGN_BOOST_MAX: float = max(
    0.0,
    min(2.0, float(os.getenv("SCALP_PING_5S_MTF_CONTINUATION_ALIGN_BOOST_MAX", "0.80"))),
)
MTF_CONTINUATION_OPPOSITE_UNITS_MULT: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_MTF_CONTINUATION_OPPOSITE_UNITS_MULT", "0.35"))),
)
MTF_CONTINUATION_BLOCK_HEAT: float = max(
    MTF_HEAT_CONTINUATION_MIN,
    min(0.99, float(os.getenv("SCALP_PING_5S_MTF_CONTINUATION_BLOCK_HEAT", "0.75"))),
)
MTF_REVERSION_TRIGGER_MULT: float = max(
    1.0, float(os.getenv("SCALP_PING_5S_MTF_REVERSION_TRIGGER_MULT", "1.35"))
)
MTF_REVERSION_IMBALANCE_MIN: float = max(
    0.5,
    min(0.99, float(os.getenv("SCALP_PING_5S_MTF_REVERSION_IMBALANCE_MIN", "0.58"))),
)
MTF_REVERSION_BOOST_MAX: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_MTF_REVERSION_BOOST_MAX", "0.35"))),
)
MTF_REGIME_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_MTF_REGIME_LOG_INTERVAL_SEC", "8.0")),
)
HORIZON_BIAS_ENABLED: bool = _bool_env("SCALP_PING_5S_HORIZON_BIAS_ENABLED", True)
HORIZON_NEUTRAL_SCORE: float = max(
    0.05,
    min(0.8, float(os.getenv("SCALP_PING_5S_HORIZON_NEUTRAL_SCORE", "0.14"))),
)
HORIZON_ALIGN_SCORE_MIN: float = max(
    HORIZON_NEUTRAL_SCORE,
    min(0.95, float(os.getenv("SCALP_PING_5S_HORIZON_ALIGN_SCORE_MIN", "0.22"))),
)
HORIZON_BLOCK_SCORE: float = max(
    HORIZON_ALIGN_SCORE_MIN,
    min(0.99, float(os.getenv("SCALP_PING_5S_HORIZON_BLOCK_SCORE", "0.44"))),
)
HORIZON_OPPOSITE_UNITS_MULT: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_HORIZON_OPPOSITE_UNITS_MULT", "0.40"))),
)
HORIZON_ALIGN_BOOST_MAX: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_HORIZON_ALIGN_BOOST_MAX", "0.32"))),
)
HORIZON_LONG_WEIGHT: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_HORIZON_LONG_WEIGHT", "0.42"))
)
HORIZON_MID_WEIGHT: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_HORIZON_MID_WEIGHT", "0.30"))
)
HORIZON_SHORT_WEIGHT: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_HORIZON_SHORT_WEIGHT", "0.18"))
)
HORIZON_MICRO_WEIGHT: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_HORIZON_MICRO_WEIGHT", "0.10"))
)
HORIZON_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_HORIZON_LOG_INTERVAL_SEC", "8.0")),
)

EXTREMA_GATE_ENABLED: bool = _bool_env("SCALP_PING_5S_EXTREMA_GATE_ENABLED", True)
EXTREMA_FAIL_OPEN: bool = _bool_env("SCALP_PING_5S_EXTREMA_FAIL_OPEN", True)
EXTREMA_M1_LOOKBACK: int = max(
    8, int(float(os.getenv("SCALP_PING_5S_EXTREMA_M1_LOOKBACK", "20")))
)
EXTREMA_M5_LOOKBACK: int = max(
    8, int(float(os.getenv("SCALP_PING_5S_EXTREMA_M5_LOOKBACK", "20")))
)
EXTREMA_H4_LOOKBACK: int = max(
    8, int(float(os.getenv("SCALP_PING_5S_EXTREMA_H4_LOOKBACK", "20")))
)
EXTREMA_M1_MIN_SPAN_PIPS: float = max(
    0.05, float(os.getenv("SCALP_PING_5S_EXTREMA_M1_MIN_SPAN_PIPS", "0.6"))
)
EXTREMA_M5_MIN_SPAN_PIPS: float = max(
    0.05, float(os.getenv("SCALP_PING_5S_EXTREMA_M5_MIN_SPAN_PIPS", "1.2"))
)
EXTREMA_H4_MIN_SPAN_PIPS: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_EXTREMA_H4_MIN_SPAN_PIPS", "6.0"))
)
EXTREMA_LONG_TOP_BLOCK_POS: float = max(
    0.50,
    min(0.99, float(os.getenv("SCALP_PING_5S_EXTREMA_LONG_TOP_BLOCK_POS", "0.86"))),
)
EXTREMA_LONG_TOP_SOFT_POS: float = max(
    0.50,
    min(
        EXTREMA_LONG_TOP_BLOCK_POS,
        float(os.getenv("SCALP_PING_5S_EXTREMA_LONG_TOP_SOFT_POS", "0.78")),
    ),
)
EXTREMA_SHORT_BOTTOM_BLOCK_POS: float = max(
    0.01,
    min(0.50, float(os.getenv("SCALP_PING_5S_EXTREMA_SHORT_BOTTOM_BLOCK_POS", "0.14"))),
)
EXTREMA_SHORT_BOTTOM_SOFT_POS: float = min(
    0.50,
    max(
        EXTREMA_SHORT_BOTTOM_BLOCK_POS,
        float(os.getenv("SCALP_PING_5S_EXTREMA_SHORT_BOTTOM_SOFT_POS", "0.22")),
    ),
)
EXTREMA_SHORT_H4_LOW_BLOCK_POS: float = max(
    0.01,
    min(0.50, float(os.getenv("SCALP_PING_5S_EXTREMA_SHORT_H4_LOW_BLOCK_POS", "0.20"))),
)
EXTREMA_SHORT_H4_LOW_SOFT_POS: float = min(
    0.60,
    max(
        EXTREMA_SHORT_H4_LOW_BLOCK_POS,
        float(os.getenv("SCALP_PING_5S_EXTREMA_SHORT_H4_LOW_SOFT_POS", "0.30")),
    ),
)
EXTREMA_REQUIRE_M1_M5_AGREE: bool = _bool_env(
    "SCALP_PING_5S_EXTREMA_REQUIRE_M1_M5_AGREE",
    True,
)
EXTREMA_SOFT_UNITS_MULT: float = max(
    0.10,
    min(1.0, float(os.getenv("SCALP_PING_5S_EXTREMA_SOFT_UNITS_MULT", "0.68"))),
)
EXTREMA_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_EXTREMA_LOG_INTERVAL_SEC", "8.0")),
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
ALLOW_OPPOSITE_WHEN_MAX_ACTIVE: bool = _bool_env(
    "SCALP_PING_5S_ALLOW_OPPOSITE_WHEN_MAX_ACTIVE",
    True,
)
ACTIVE_CAP_MARGIN_BYPASS_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_ACTIVE_CAP_MARGIN_BYPASS_ENABLED",
    False,
)
ACTIVE_CAP_BYPASS_MIN_FREE_RATIO: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_ACTIVE_CAP_BYPASS_MIN_FREE_RATIO", "0.06")),
)
ACTIVE_CAP_BYPASS_MIN_MARGIN_AVAILABLE_JPY: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_ACTIVE_CAP_BYPASS_MIN_MARGIN_AVAILABLE_JPY", "8000")),
)
ACTIVE_CAP_BYPASS_EXTRA_TOTAL: int = max(
    0,
    int(float(os.getenv("SCALP_PING_5S_ACTIVE_CAP_BYPASS_EXTRA_TOTAL", "12"))),
)
ACTIVE_CAP_BYPASS_EXTRA_PER_DIRECTION: int = max(
    0,
    int(float(os.getenv("SCALP_PING_5S_ACTIVE_CAP_BYPASS_EXTRA_PER_DIRECTION", "8"))),
)
ACTIVE_CAP_BYPASS_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_ACTIVE_CAP_BYPASS_LOG_INTERVAL_SEC", "8.0")),
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
LOW_MARGIN_HEDGE_RELIEF_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_LOW_MARGIN_HEDGE_RELIEF_ENABLED",
    False,
)
LOW_MARGIN_HEDGE_RELIEF_MIN_FREE_RATIO: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_LOW_MARGIN_HEDGE_RELIEF_MIN_FREE_RATIO", "0.015")),
)
LOW_MARGIN_HEDGE_RELIEF_MIN_MARGIN_AVAILABLE_JPY: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_LOW_MARGIN_HEDGE_RELIEF_MIN_MARGIN_AVAILABLE_JPY",
            "1500",
        )
    ),
)
LOW_MARGIN_HEDGE_RELIEF_MAX_IMBALANCE_FRACTION: float = max(
    0.1,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_LOW_MARGIN_HEDGE_RELIEF_MAX_IMBALANCE_FRACTION",
                "0.60",
            )
        ),
    ),
)
LOW_MARGIN_HEDGE_RELIEF_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_LOW_MARGIN_HEDGE_RELIEF_LOG_INTERVAL_SEC", "8.0")),
)

CONFIDENCE_FLOOR: int = max(0, int(float(os.getenv("SCALP_PING_5S_CONF_FLOOR", "58"))))
CONFIDENCE_CEIL: int = max(
    CONFIDENCE_FLOOR + 1, int(float(os.getenv("SCALP_PING_5S_CONF_CEIL", "92")))
)

SIDE_BIAS_ENABLED: bool = _bool_env("SCALP_PING_5S_SIDE_BIAS_ENABLED", True)
SIDE_BIAS_WINDOW_SEC: float = max(
    SIGNAL_WINDOW_SEC,
    float(os.getenv("SCALP_PING_5S_SIDE_BIAS_WINDOW_SEC", "18.0")),
)
SIDE_BIAS_MIN_TICKS: int = max(
    2,
    int(float(os.getenv("SCALP_PING_5S_SIDE_BIAS_MIN_TICKS", str(MIN_SIGNAL_TICKS)))),
)
SIDE_BIAS_MIN_DRIFT_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_SIDE_BIAS_MIN_DRIFT_PIPS", "0.6"))
)
SIDE_BIAS_SCALE_GAIN: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_SIDE_BIAS_SCALE_GAIN", "0.35"))
)
SIDE_BIAS_SCALE_FLOOR: float = max(
    0.1, min(1.0, float(os.getenv("SCALP_PING_5S_SIDE_BIAS_SCALE_FLOOR", "0.35")))
)
SIDE_BIAS_BLOCK_THRESHOLD: float = max(
    0.0, min(1.0, float(os.getenv("SCALP_PING_5S_SIDE_BIAS_BLOCK_THRESHOLD", "0.0")))
)

TP_BASE_PIPS: float = max(0.2, float(os.getenv("SCALP_PING_5S_TP_BASE_PIPS", "0.2")))
TP_NET_MIN_PIPS: float = max(0.1, float(os.getenv("SCALP_PING_5S_TP_NET_MIN_PIPS", "0.25")))
TP_MAX_PIPS: float = max(TP_BASE_PIPS, float(os.getenv("SCALP_PING_5S_TP_MAX_PIPS", "1.0")))
TP_MOMENTUM_BONUS_MAX: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_TP_MOMENTUM_BONUS_MAX", "0.2"))
)
TP_TIME_ADAPT_ENABLED: bool = _bool_env("SCALP_PING_5S_TP_TIME_ADAPT_ENABLED", True)
TP_TARGET_HOLD_SEC: float = max(
    5.0, float(os.getenv("SCALP_PING_5S_TP_TARGET_HOLD_SEC", "120.0"))
)
TP_HOLD_LOOKBACK_HOURS: float = max(
    1.0, float(os.getenv("SCALP_PING_5S_TP_HOLD_LOOKBACK_HOURS", "24.0"))
)
TP_HOLD_MIN_TRADES: int = max(
    5, int(float(os.getenv("SCALP_PING_5S_TP_HOLD_MIN_TRADES", "30")))
)
TP_HOLD_STATS_TTL_SEC: float = max(
    5.0, float(os.getenv("SCALP_PING_5S_TP_HOLD_STATS_TTL_SEC", "30.0"))
)
TP_TIME_MULT_MIN: float = max(
    0.1, min(1.0, float(os.getenv("SCALP_PING_5S_TP_TIME_MULT_MIN", "0.55")))
)
TP_TIME_MULT_MAX: float = max(
    TP_TIME_MULT_MIN,
    min(1.2, float(os.getenv("SCALP_PING_5S_TP_TIME_MULT_MAX", "1.0"))),
)
TP_NET_MIN_FLOOR_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_TP_NET_MIN_FLOOR_PIPS", "0.10"))
)
ENTRY_CHASE_MAX_PIPS: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_ENTRY_CHASE_MAX_PIPS", "1.4"))
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
ENTRY_QUALITY_WINDOW_SEC: float = max(
    1.0,
    float(os.getenv("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_WINDOW_SEC", "30.0")),
)
ENTRY_QUALITY_MAX_AGE_MS: float = max(
    0.0,
    float(os.getenv("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MAX_AGE_MS", "2500.0")),
)
ENTRY_QUALITY_MIN_SPAN_RATIO: float = max(
    0.0,
    min(1.0, float(os.getenv("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_SPAN_RATIO", "0.7"))),
)
ENTRY_QUALITY_MIN_DENSITY: float = max(
    0.0,
    float(
        os.getenv(
            f"ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_TICK_DENSITY_{POCKET.upper()}",
            os.getenv("ORDER_ENTRY_QUALITY_MICROSTRUCTURE_MIN_TICK_DENSITY", "0.0"),
        )
    ),
)
SNAPSHOT_TOPUP_ENABLED: bool = _bool_env("SCALP_PING_5S_SNAPSHOT_TOPUP_ENABLED", True)
SNAPSHOT_TOPUP_MIN_INTERVAL_SEC: float = max(
    SNAPSHOT_MIN_INTERVAL_SEC,
    float(os.getenv("SCALP_PING_5S_SNAPSHOT_TOPUP_MIN_INTERVAL_SEC", "1.0")),
)
SNAPSHOT_TOPUP_TARGET_DENSITY: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_SNAPSHOT_TOPUP_TARGET_DENSITY",
            str(ENTRY_QUALITY_MIN_DENSITY),
        )
    ),
)
SNAPSHOT_KEEPALIVE_ENABLED: bool = _bool_env("SCALP_PING_5S_SNAPSHOT_KEEPALIVE_ENABLED", True)
SNAPSHOT_KEEPALIVE_MIN_INTERVAL_SEC: float = max(
    SNAPSHOT_MIN_INTERVAL_SEC,
    float(os.getenv("SCALP_PING_5S_SNAPSHOT_KEEPALIVE_MIN_INTERVAL_SEC", "0.8")),
)
SNAPSHOT_KEEPALIVE_MAX_AGE_MS: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_SNAPSHOT_KEEPALIVE_MAX_AGE_MS",
            str(ENTRY_QUALITY_MAX_AGE_MS),
        )
    ),
)
SNAPSHOT_KEEPALIVE_MIN_DENSITY: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_SNAPSHOT_KEEPALIVE_MIN_DENSITY",
            str(ENTRY_QUALITY_MIN_DENSITY),
        )
    ),
)
SNAPSHOT_KEEPALIVE_MIN_SPAN_RATIO: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SNAPSHOT_KEEPALIVE_MIN_SPAN_RATIO",
                str(ENTRY_QUALITY_MIN_SPAN_RATIO),
            )
        ),
    ),
)

FORCE_EXIT_MAX_HOLD_SEC: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_HOLD_SEC", "0.0"))
)
FORCE_EXIT_MAX_FLOATING_LOSS_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS", "0.0"))
)
FORCE_EXIT_RECOVERY_WINDOW_SEC: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_RECOVERY_WINDOW_SEC", "0.0"))
)
FORCE_EXIT_RECOVERABLE_LOSS_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_RECOVERABLE_LOSS_PIPS", "0.0"))
)
FORCE_EXIT_MAX_ACTIONS: int = max(
    1, int(float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_ACTIONS", "3")))
)
FORCE_EXIT_REASON: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_REASON", "time_stop").strip() or "time_stop"
)
FORCE_EXIT_MAX_FLOATING_LOSS_REASON: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_FLOATING_LOSS_REASON", "max_floating_loss").strip()
    or "max_floating_loss"
)
FORCE_EXIT_RECOVERY_REASON: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_RECOVERY_REASON", "no_recovery").strip()
    or "no_recovery"
)
FORCE_EXIT_GIVEBACK_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_FORCE_EXIT_GIVEBACK_ENABLED",
    False,
)
FORCE_EXIT_GIVEBACK_ARM_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_ARM_PIPS", "0.0"))
)
FORCE_EXIT_GIVEBACK_BACKOFF_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_BACKOFF_PIPS", "0.0"))
)
FORCE_EXIT_GIVEBACK_MIN_HOLD_SEC: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_MIN_HOLD_SEC", "0.0"))
)
FORCE_EXIT_GIVEBACK_PROTECT_PIPS: float = float(
    os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_PROTECT_PIPS", "0.0")
)
FORCE_EXIT_GIVEBACK_REASON: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_GIVEBACK_REASON", "giveback_lock").strip()
    or "giveback_lock"
)
FORCE_EXIT_REQUIRE_POLICY_GENERATION: bool = _bool_env(
    "SCALP_PING_5S_FORCE_EXIT_REQUIRE_POLICY_GENERATION",
    True,
)
FORCE_EXIT_POLICY_GENERATION: str = (
    os.getenv(
        "SCALP_PING_5S_FORCE_EXIT_POLICY_GENERATION",
        os.getenv("ORDER_ENTRY_POLICY_GENERATION", ""),
    ).strip()
)
FORCE_EXIT_SKIP_EXISTING_ON_START: bool = _bool_env(
    "SCALP_PING_5S_FORCE_EXIT_SKIP_EXISTING_ON_START",
    True,
)
FORCE_EXIT_MTF_FIB_HOLD_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_FORCE_EXIT_MTF_FIB_HOLD_ENABLED",
    True,
)
FORCE_EXIT_MTF_FIB_LOOKBACK_M5: int = max(
    12,
    int(float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_LOOKBACK_M5", "72"))),
)
FORCE_EXIT_MTF_FIB_LOOKBACK_H1: int = max(
    8,
    int(float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_LOOKBACK_H1", "36"))),
)
FORCE_EXIT_MTF_FIB_LOWER: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_LOWER", "0.382"))),
)
FORCE_EXIT_MTF_FIB_UPPER: float = max(
    FORCE_EXIT_MTF_FIB_LOWER,
    min(1.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_UPPER", "0.618"))),
)
FORCE_EXIT_MTF_FIB_MIN_RANGE_PIPS: float = max(
    0.5,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_MIN_RANGE_PIPS", "6.0")),
)
FORCE_EXIT_MTF_FIB_MAX_WAIT_SEC: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_MAX_WAIT_SEC", "180.0")),
)
FORCE_EXIT_MTF_FIB_MAX_HOLD_LOSS_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_MAX_HOLD_LOSS_PIPS", "3.8")),
)
FORCE_EXIT_MTF_FIB_MIN_RECOVER_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_MIN_RECOVER_PIPS", "0.45")),
)
FORCE_EXIT_MTF_FIB_PROJECTED_MAX_LOSS_PIPS: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_FORCE_EXIT_MTF_FIB_PROJECTED_MAX_LOSS_PIPS",
            "1.6",
        )
    ),
)
FORCE_EXIT_MTF_FIB_MAX_TARGET_PIPS: float = max(
    FORCE_EXIT_MTF_FIB_MIN_RECOVER_PIPS,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_MAX_TARGET_PIPS", "2.6")),
)
FORCE_EXIT_MTF_FIB_OPPOSITE_HEAT_BLOCK: float = max(
    0.0,
    min(
        1.0,
        float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_OPPOSITE_HEAT_BLOCK", "0.72")),
    ),
)
FORCE_EXIT_MTF_FIB_OPPOSITE_RSI: float = max(
    1.0,
    min(99.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_OPPOSITE_RSI", "42.0"))),
)
FORCE_EXIT_MTF_FIB_OPPOSITE_EMA_GAP_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_OPPOSITE_EMA_GAP_PIPS", "0.35")),
)
FORCE_EXIT_MTF_FIB_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MTF_FIB_LOG_INTERVAL_SEC", "10.0")),
)

PROFIT_BANK_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_PROFIT_BANK_ENABLED",
    False,
)
PROFIT_BANK_START_TIME_UTC: str = (
    os.getenv("SCALP_PING_5S_PROFIT_BANK_START_TIME_UTC", "").strip()
)
PROFIT_BANK_REASON: str = (
    os.getenv("SCALP_PING_5S_PROFIT_BANK_REASON", "profit_bank_release").strip()
    or "profit_bank_release"
)
PROFIT_BANK_MIN_GROSS_PROFIT_JPY: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_PROFIT_BANK_MIN_GROSS_PROFIT_JPY", "200.0")),
)
PROFIT_BANK_MIN_NET_KEEP_JPY: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_PROFIT_BANK_MIN_NET_KEEP_JPY", "120.0")),
)
PROFIT_BANK_SPEND_RATIO: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_PROFIT_BANK_SPEND_RATIO", "0.35"))),
)
PROFIT_BANK_MIN_BUFFER_JPY: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_PROFIT_BANK_MIN_BUFFER_JPY", "80.0")),
)
PROFIT_BANK_MIN_TARGET_LOSS_JPY: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_PROFIT_BANK_MIN_TARGET_LOSS_JPY", "60.0")),
)
PROFIT_BANK_MAX_TARGET_LOSS_JPY: float = max(
    PROFIT_BANK_MIN_TARGET_LOSS_JPY,
    float(os.getenv("SCALP_PING_5S_PROFIT_BANK_MAX_TARGET_LOSS_JPY", "1500.0")),
)
PROFIT_BANK_TARGET_MIN_HOLD_SEC: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_PROFIT_BANK_TARGET_MIN_HOLD_SEC", "120.0")),
)
PROFIT_BANK_TARGET_ORDER: str = (
    os.getenv("SCALP_PING_5S_PROFIT_BANK_TARGET_ORDER", "largest_loss").strip().lower()
    or "largest_loss"
)
if PROFIT_BANK_TARGET_ORDER not in {"largest_loss", "oldest"}:
    PROFIT_BANK_TARGET_ORDER = "largest_loss"
PROFIT_BANK_TARGET_REQUIRE_OPEN_BEFORE_START: bool = _bool_env(
    "SCALP_PING_5S_PROFIT_BANK_TARGET_REQUIRE_OPEN_BEFORE_START",
    False,
)
PROFIT_BANK_MAX_ACTIONS: int = max(
    1,
    int(float(os.getenv("SCALP_PING_5S_PROFIT_BANK_MAX_ACTIONS", "1"))),
)
PROFIT_BANK_COOLDOWN_SEC: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_PROFIT_BANK_COOLDOWN_SEC", "15.0")),
)
PROFIT_BANK_STATS_TTL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_PROFIT_BANK_STATS_TTL_SEC", "6.0")),
)
PROFIT_BANK_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_PROFIT_BANK_LOG_INTERVAL_SEC", "12.0")),
)
PROFIT_BANK_EXCLUDE_TRADE_IDS: tuple[str, ...] = _csv_env(
    "SCALP_PING_5S_PROFIT_BANK_EXCLUDE_TRADE_IDS"
)
PROFIT_BANK_EXCLUDE_CLIENT_IDS: tuple[str, ...] = _csv_env(
    "SCALP_PING_5S_PROFIT_BANK_EXCLUDE_CLIENT_IDS"
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
