from __future__ import annotations

import os

from execution.stop_loss_policy import stop_loss_disabled_for_pocket


ENV_PREFIX = (
    os.getenv("SCALP_PING_5S_ENV_PREFIX", "SCALP_PING_5S").strip()
    or "SCALP_PING_5S"
)
_IS_B_OR_C_PREFIX = ENV_PREFIX in {"SCALP_PING_5S_B", "SCALP_PING_5S_C", "SCALP_PING_5S_D"}
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


def _csv_float_env(name: str, default: str) -> tuple[float, ...]:
    raw = str(os.getenv(name, default) or "").strip()
    if not raw:
        return ()
    values: list[float] = []
    for part in raw.replace("\n", ",").split(","):
        token = str(part).strip()
        if not token:
            continue
        try:
            parsed = float(token)
        except ValueError:
            continue
        if parsed > 0.0:
            values.append(parsed)
    deduped: list[float] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _csv_int_env(name: str, default: str = "") -> tuple[int, ...]:
    raw = str(os.getenv(name, default) or "").strip()
    if not raw:
        return ()
    values: list[int] = []
    for part in raw.replace("\n", ",").split(","):
        token = str(part).strip()
        if not token:
            continue
        try:
            parsed = int(float(token))
        except ValueError:
            continue
        parsed = parsed % 24
        if parsed not in values:
            values.append(parsed)
    return tuple(values)


def _normalize_side(raw: str) -> str:
    side = str(raw or "").strip().lower()
    if side in {"buy", "long", "open_long"}:
        return "long"
    if side in {"sell", "short", "open_short"}:
        return "short"
    return ""


ENABLED: bool = _bool_env("SCALP_PING_5S_ENABLED", False)
REQUIRE_PRACTICE: bool = _bool_env("SCALP_PING_5S_REQUIRE_PRACTICE", True)
POCKET: str = os.getenv("SCALP_PING_5S_POCKET", "scalp_fast").strip() or "scalp_fast"
STRATEGY_TAG: str = (
    os.getenv("SCALP_PING_5S_STRATEGY_TAG", "scalp_ping_5s").strip() or "scalp_ping_5s"
)
LOG_PREFIX: str = os.getenv("SCALP_PING_5S_LOG_PREFIX", "[SCALP_PING_5S]")
PATTERN_GATE_OPT_IN: bool = _bool_env("SCALP_PING_5S_PATTERN_GATE_OPT_IN", True)
SIDE_FILTER: str = _normalize_side(os.getenv("SCALP_PING_5S_SIDE_FILTER", ""))
DROP_FLOW_ONLY: bool = _bool_env("SCALP_PING_5S_DROP_FLOW_ONLY", False)
DROP_FLOW_WINDOW_SEC: float = max(0.5, float(os.getenv("SCALP_PING_5S_DROP_FLOW_WINDOW_SEC", "15.0")))
DROP_FLOW_MIN_PIPS: float = max(0.05, float(os.getenv("SCALP_PING_5S_DROP_FLOW_MIN_PIPS", "0.30")))
DROP_FLOW_MIN_TICKS: int = max(2, int(float(os.getenv("SCALP_PING_5S_DROP_FLOW_MIN_TICKS", "6"))))
DROP_FLOW_MAX_BOUNCE_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_DROP_FLOW_MAX_BOUNCE_PIPS", "0.05")),
)
BLOCK_HOURS_JST: tuple[int, ...] = _csv_int_env("SCALP_PING_5S_BLOCK_HOURS_JST")
ALLOW_HOURS_JST: tuple[int, ...] = _csv_int_env("SCALP_PING_5S_ALLOW_HOURS_JST")

LOOP_INTERVAL_SEC: float = max(0.05, float(os.getenv("SCALP_PING_5S_LOOP_INTERVAL_SEC", "0.2")))
WINDOW_SEC: float = max(2.0, float(os.getenv("SCALP_PING_5S_WINDOW_SEC", "5.0")))
SIGNAL_WINDOW_SEC: float = max(0.4, float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_SEC", "1.2")))
SIGNAL_WINDOW_FALLBACK_SEC: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_SEC", "0.0")),
)
SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW: bool = _bool_env(
    "SCALP_PING_5S_SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW",
    True,
)
SIGNAL_WINDOW_ADAPTIVE_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_ENABLED",
    False,
)
SIGNAL_WINDOW_ADAPTIVE_SHADOW_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_SHADOW_ENABLED",
    False,
)
SIGNAL_WINDOW_ADAPTIVE_CANDIDATES_SEC: tuple[float, ...] = _csv_float_env(
    "SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_CANDIDATES_SEC",
    "0.4,0.78,1.2,1.5,2.0,2.857,8.0",
)
SIGNAL_WINDOW_ADAPTIVE_SCALE_WITH_SPEED: bool = _bool_env(
    "SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_SCALE_WITH_SPEED",
    True,
)
SIGNAL_WINDOW_ADAPTIVE_MIN_SEC: float = max(
    0.2,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_MIN_SEC", "0.3")),
)
SIGNAL_WINDOW_ADAPTIVE_MAX_SEC: float = max(
    SIGNAL_WINDOW_ADAPTIVE_MIN_SEC + 0.1,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_MAX_SEC", str(WINDOW_SEC))),
)
SIGNAL_WINDOW_ADAPTIVE_STATS_TTL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_STATS_TTL_SEC", "25.0")),
)
SIGNAL_WINDOW_ADAPTIVE_LOOKBACK_HOURS: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_LOOKBACK_HOURS", "48.0")),
)
SIGNAL_WINDOW_ADAPTIVE_MIN_TRADES: int = max(
    5,
    int(float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_MIN_TRADES", "30"))),
)
SIGNAL_WINDOW_ADAPTIVE_MATCH_TOL_SEC: float = max(
    0.01,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_MATCH_TOL_SEC", "0.35")),
)
SIGNAL_WINDOW_ADAPTIVE_BUCKET_SEC: float = max(
    0.01,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_BUCKET_SEC", "0.05")),
)
SIGNAL_WINDOW_ADAPTIVE_UCB_BONUS_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_UCB_BONUS_PIPS", "0.12")),
)
SIGNAL_WINDOW_ADAPTIVE_COLDSTART_PENALTY_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_COLDSTART_PENALTY_PIPS", "0.20")),
)
SIGNAL_WINDOW_ADAPTIVE_SELECTION_MARGIN_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_SELECTION_MARGIN_PIPS", "0.05")),
)
SIGNAL_WINDOW_ADAPTIVE_SHADOW_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_SIGNAL_WINDOW_ADAPTIVE_SHADOW_LOG_INTERVAL_SEC", "15.0")),
)
INSTANT_VOL_WINDOW_SEC: float = max(
    0.25,
    float(os.getenv("SCALP_PING_5S_INSTANT_VOL_WINDOW_SEC", "0.55")),
)
INSTANT_VOL_BUCKET_WEIGHT: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_INSTANT_VOL_BUCKET_WEIGHT", "0.55"))),
)
INSTANT_VOL_ADAPT_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_INSTANT_VOL_ADAPT_ENABLED",
    True,
)
INSTANT_SPEED_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_INSTANT_SPEED_ENABLED",
    True,
)
INSTANT_SPEED_VOL_LOW_PIPS: float = max(
    0.05,
    float(os.getenv("SCALP_PING_5S_INSTANT_SPEED_VOL_LOW_PIPS", "0.45")),
)
INSTANT_SPEED_VOL_HIGH_PIPS: float = max(
    INSTANT_SPEED_VOL_LOW_PIPS + 0.1,
    float(os.getenv("SCALP_PING_5S_INSTANT_SPEED_VOL_HIGH_PIPS", "1.35")),
)
INSTANT_SPEED_SCALE_MIN: float = max(
    0.2,
    min(1.0, float(os.getenv("SCALP_PING_5S_INSTANT_SPEED_SCALE_MIN", "0.55"))),
)
INSTANT_LOOKAHEAD_SCALE_MIN: float = max(
    INSTANT_SPEED_SCALE_MIN,
    min(1.0, float(os.getenv("SCALP_PING_5S_INSTANT_LOOKAHEAD_SCALE_MIN", "0.35"))),
)
INSTANT_COOLDOWN_SCALE_MIN: float = max(
    INSTANT_SPEED_SCALE_MIN,
    min(1.0, float(os.getenv("SCALP_PING_5S_INSTANT_COOLDOWN_SCALE_MIN", "0.35"))),
)
MIN_TICKS: int = max(4, int(float(os.getenv("SCALP_PING_5S_MIN_TICKS", "12"))))
MIN_SIGNAL_TICKS: int = max(3, int(float(os.getenv("SCALP_PING_5S_MIN_SIGNAL_TICKS", "5"))))
LONG_MIN_SIGNAL_TICKS: int = max(
    3,
    int(float(os.getenv("SCALP_PING_5S_LONG_MIN_SIGNAL_TICKS", str(MIN_SIGNAL_TICKS)))),
)
SHORT_MIN_SIGNAL_TICKS: int = max(
    3,
    int(float(os.getenv("SCALP_PING_5S_SHORT_MIN_SIGNAL_TICKS", str(MIN_SIGNAL_TICKS)))),
)
MIN_TICK_RATE: float = max(0.5, float(os.getenv("SCALP_PING_5S_MIN_TICK_RATE", "4.0")))
LONG_MIN_TICK_RATE: float = max(
    0.5,
    float(os.getenv("SCALP_PING_5S_LONG_MIN_TICK_RATE", str(MIN_TICK_RATE))),
)
SHORT_MIN_TICK_RATE: float = max(
    0.5,
    float(os.getenv("SCALP_PING_5S_SHORT_MIN_TICK_RATE", str(MIN_TICK_RATE))),
)
MAX_TICK_AGE_MS: float = max(100.0, float(os.getenv("SCALP_PING_5S_MAX_TICK_AGE_MS", "800.0")))

MAX_SPREAD_PIPS: float = max(0.1, float(os.getenv("SCALP_PING_5S_MAX_SPREAD_PIPS", "0.85")))
MOMENTUM_TRIGGER_PIPS: float = max(0.1, float(os.getenv("SCALP_PING_5S_MOMENTUM_TRIGGER_PIPS", "0.8")))
LONG_MOMENTUM_TRIGGER_PIPS: float = max(
    0.1,
    float(
        os.getenv(
            "SCALP_PING_5S_LONG_MOMENTUM_TRIGGER_PIPS",
            str(MOMENTUM_TRIGGER_PIPS),
        )
    ),
)
SHORT_MOMENTUM_TRIGGER_PIPS: float = max(
    0.1,
    float(
        os.getenv(
            "SCALP_PING_5S_SHORT_MOMENTUM_TRIGGER_PIPS",
            str(MOMENTUM_TRIGGER_PIPS),
        )
    ),
)
MOMENTUM_SPREAD_MULT: float = max(0.0, float(os.getenv("SCALP_PING_5S_MOMENTUM_SPREAD_MULT", "1.0")))
ENTRY_BID_ASK_EDGE_PIPS: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_ENTRY_BID_ASK_EDGE_PIPS",
            "0.12",
        )
    ),
)
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
DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT: float = max(
    0.05,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT",
                str(DIRECTION_BIAS_OPPOSITE_UNITS_MULT),
            )
        )
    )
)
DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT: float = max(
    0.05,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT",
                str(DIRECTION_BIAS_OPPOSITE_UNITS_MULT),
            )
        )
    )
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
FAST_DIRECTION_FLIP_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_FAST_DIRECTION_FLIP_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN: float = max(
    DIRECTION_BIAS_NEUTRAL_SCORE,
    min(
        0.95,
        float(
            os.getenv(
                "SCALP_PING_5S_FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN",
                "0.44",
            )
        ),
    ),
)
FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN: float = max(
    DIRECTION_BIAS_NEUTRAL_SCORE,
    min(
        0.95,
        float(
            os.getenv(
                "SCALP_PING_5S_FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN",
                "0.24",
            )
        ),
    ),
)
FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN: int = max(
    1,
    int(float(os.getenv("SCALP_PING_5S_FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN", "2"))),
)
FAST_DIRECTION_FLIP_NEUTRAL_HORIZON_BIAS_SCORE_MIN: float = max(
    FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_FAST_DIRECTION_FLIP_NEUTRAL_HORIZON_BIAS_SCORE_MIN",
                "0.70",
            )
        ),
    ),
)
FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS", "0.08")),
)
FAST_DIRECTION_FLIP_CONFIDENCE_ADD: int = max(
    0,
    int(float(os.getenv("SCALP_PING_5S_FAST_DIRECTION_FLIP_CONFIDENCE_ADD", "3"))),
)
FAST_DIRECTION_FLIP_COOLDOWN_SEC: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_FAST_DIRECTION_FLIP_COOLDOWN_SEC", "0.8")),
)
FAST_DIRECTION_FLIP_REGIME_BLOCK_SCORE: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_FAST_DIRECTION_FLIP_REGIME_BLOCK_SCORE",
                "0.62",
            )
        ),
    ),
)
SL_STREAK_DIRECTION_FLIP_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
SL_STREAK_DIRECTION_FLIP_MIN_STREAK: int = max(
    1,
    int(float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_MIN_STREAK", "2"))),
)
SL_STREAK_DIRECTION_FLIP_LOOKBACK_TRADES: int = max(
    SL_STREAK_DIRECTION_FLIP_MIN_STREAK,
    int(float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_LOOKBACK_TRADES", "6"))),
)
SL_STREAK_DIRECTION_FLIP_MAX_AGE_SEC: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_MAX_AGE_SEC", "180.0")),
)
SL_STREAK_DIRECTION_FLIP_CONFIDENCE_ADD: int = max(
    0,
    int(float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_CONFIDENCE_ADD", "4"))),
)
SL_STREAK_DIRECTION_FLIP_CACHE_TTL_SEC: float = max(
    0.1,
    float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_CACHE_TTL_SEC", "1.0")),
)
SL_STREAK_DIRECTION_FLIP_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_LOG_INTERVAL_SEC", "6.0")),
)
SL_STREAK_DIRECTION_FLIP_ALLOW_WITH_FAST_FLIP: bool = _bool_env(
    "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_ALLOW_WITH_FAST_FLIP",
    False,
)
SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS: int = max(
    1,
    int(float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS", "2"))),
)
SL_STREAK_DIRECTION_FLIP_MIN_TARGET_MARKET_PLUS: int = max(
    0,
    int(float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_MIN_TARGET_MARKET_PLUS", "1"))),
)
SL_STREAK_DIRECTION_FLIP_METRICS_OVERRIDE_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_METRICS_OVERRIDE_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
SL_STREAK_DIRECTION_FLIP_METRICS_SIDE_TRADES_MIN: int = max(
    1,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_METRICS_SIDE_TRADES_MIN",
                "4" if _IS_B_OR_C_PREFIX else "8",
            )
        )
    ),
)
SL_STREAK_DIRECTION_FLIP_METRICS_SIDE_SL_RATE_MIN: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_METRICS_SIDE_SL_RATE_MIN",
                "0.50" if _IS_B_OR_C_PREFIX else "0.60",
            )
        ),
    ),
)
SL_STREAK_DIRECTION_FLIP_FORCE_STREAK: int = max(
    SL_STREAK_DIRECTION_FLIP_MIN_STREAK,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_FORCE_STREAK",
                (
                    "3"
                    if _IS_B_OR_C_PREFIX
                    else str(SL_STREAK_DIRECTION_FLIP_MIN_STREAK + 2)
                ),
            )
        )
    ),
)
SL_STREAK_DIRECTION_FLIP_METRICS_LOOKBACK_TRADES: int = max(
    SL_STREAK_DIRECTION_FLIP_LOOKBACK_TRADES,
    int(float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_METRICS_LOOKBACK_TRADES", "24"))),
)
SL_STREAK_DIRECTION_FLIP_METRICS_CACHE_TTL_SEC: float = max(
    0.1,
    float(os.getenv("SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_METRICS_CACHE_TTL_SEC", "2.0")),
)
SL_STREAK_DIRECTION_FLIP_REQUIRE_TECH_CONFIRM: bool = _bool_env(
    "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_REQUIRE_TECH_CONFIRM",
    True if _IS_B_OR_C_PREFIX else False,
)
SL_STREAK_DIRECTION_FLIP_FORCE_WITHOUT_TECH_CONFIRM: bool = _bool_env(
    "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_FORCE_WITHOUT_TECH_CONFIRM",
    True if _IS_B_OR_C_PREFIX else False,
)
SL_STREAK_DIRECTION_FLIP_DIRECTION_SCORE_MIN: float = max(
    DIRECTION_BIAS_NEUTRAL_SCORE,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_DIRECTION_SCORE_MIN",
                "0.40",
            )
        ),
    ),
)
SL_STREAK_DIRECTION_FLIP_HORIZON_SCORE_MIN: float = max(
    DIRECTION_BIAS_NEUTRAL_SCORE,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SL_STREAK_DIRECTION_FLIP_HORIZON_SCORE_MIN",
                "0.24",
            )
        ),
    ),
)
SIDE_METRICS_DIRECTION_FLIP_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
SIDE_METRICS_DIRECTION_FLIP_CACHE_TTL_SEC: float = max(
    0.1,
    float(
        os.getenv(
            "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_CACHE_TTL_SEC",
            "1.0",
        )
    ),
)
SIDE_METRICS_DIRECTION_FLIP_LOOKBACK_TRADES: int = max(
    4,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_LOOKBACK_TRADES",
                "36",
            )
        )
    ),
)
SIDE_METRICS_DIRECTION_FLIP_MIN_CURRENT_TRADES: int = max(
    1,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_MIN_CURRENT_TRADES",
                "8",
            )
        )
    ),
)
SIDE_METRICS_DIRECTION_FLIP_MIN_TARGET_TRADES: int = max(
    1,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_MIN_TARGET_TRADES",
                "6",
            )
        )
    ),
)
SIDE_METRICS_DIRECTION_FLIP_MIN_CURRENT_SL_RATE: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_MIN_CURRENT_SL_RATE",
                "0.62",
            )
        ),
    ),
)
SIDE_METRICS_DIRECTION_FLIP_MIN_SL_GAP: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_MIN_SL_GAP",
                "0.22",
            )
        ),
    ),
)
SIDE_METRICS_DIRECTION_FLIP_MIN_MARKET_PLUS_GAP: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_MIN_MARKET_PLUS_GAP",
                "0.12",
            )
        ),
    ),
)
SIDE_METRICS_DIRECTION_FLIP_CONFIDENCE_ADD: int = max(
    0,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_CONFIDENCE_ADD",
                "2",
            )
        )
    ),
)
SIDE_METRICS_DIRECTION_FLIP_COOLDOWN_SEC: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_COOLDOWN_SEC",
            "0.8",
        )
    ),
)
SIDE_METRICS_DIRECTION_FLIP_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(
        os.getenv(
            "SCALP_PING_5S_SIDE_METRICS_DIRECTION_FLIP_LOG_INTERVAL_SEC",
            "6.0",
        )
    ),
)
SIDE_ADVERSE_STACK_UNITS_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
SIDE_ADVERSE_STACK_UNITS_CACHE_TTL_SEC: float = max(
    0.1,
    float(
        os.getenv(
            "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_CACHE_TTL_SEC",
            "1.0",
        )
    ),
)
SIDE_ADVERSE_STACK_UNITS_LOOKBACK_TRADES: int = max(
    4,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_LOOKBACK_TRADES",
                "36",
            )
        )
    ),
)
SIDE_ADVERSE_STACK_UNITS_MIN_CURRENT_TRADES: int = max(
    1,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_MIN_CURRENT_TRADES",
                "8",
            )
        )
    ),
)
SIDE_ADVERSE_STACK_UNITS_MIN_TARGET_TRADES: int = max(
    1,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_MIN_TARGET_TRADES",
                "6",
            )
        )
    ),
)
SIDE_ADVERSE_STACK_UNITS_MIN_CURRENT_SL_RATE: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_MIN_CURRENT_SL_RATE",
                "0.58",
            )
        ),
    ),
)
SIDE_ADVERSE_STACK_UNITS_MIN_SL_GAP: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_MIN_SL_GAP",
                "0.16",
            )
        ),
    ),
)
SIDE_ADVERSE_STACK_UNITS_MIN_MARKET_PLUS_GAP: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_MIN_MARKET_PLUS_GAP",
                "0.08",
            )
        ),
    ),
)
SIDE_ADVERSE_STACK_UNITS_ACTIVE_START: int = max(
    1,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_ACTIVE_START",
                "3",
            )
        )
    ),
)
SIDE_ADVERSE_STACK_UNITS_STEP_MULT: float = max(
    0.0,
    min(
        0.9,
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_STEP_MULT",
                "0.14",
            )
        ),
    ),
)
SIDE_ADVERSE_STACK_UNITS_MIN_MULT: float = max(
    0.05,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_SIDE_ADVERSE_STACK_UNITS_MIN_MULT",
                "0.24",
            )
        ),
    ),
)
SIDE_ADVERSE_STACK_DD_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_SIDE_ADVERSE_STACK_DD_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
SIDE_ADVERSE_STACK_DD_START_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_SIDE_ADVERSE_STACK_DD_START_PIPS", "0.60")),
)
SIDE_ADVERSE_STACK_DD_FULL_PIPS: float = max(
    SIDE_ADVERSE_STACK_DD_START_PIPS + 0.05,
    float(os.getenv("SCALP_PING_5S_SIDE_ADVERSE_STACK_DD_FULL_PIPS", "2.20")),
)
SIDE_ADVERSE_STACK_DD_MIN_MULT: float = max(
    0.05,
    min(
        1.0,
        float(os.getenv("SCALP_PING_5S_SIDE_ADVERSE_STACK_DD_MIN_MULT", "0.30")),
    ),
)
SIDE_ADVERSE_STACK_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_SIDE_ADVERSE_STACK_LOG_INTERVAL_SEC", "6.0")),
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
REVERT_DIRECTION_OPPOSITE_BLOCK_MIN_MULT: float = max(
    0.1,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_REVERT_DIRECTION_OPPOSITE_BLOCK_MIN_MULT",
                "0.45",
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
MTF_CONTINUATION_OPPOSITE_BLOCK_MIN_MULT: float = max(
    0.05,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_MTF_CONTINUATION_OPPOSITE_BLOCK_MIN_MULT",
                "0.18",
            )
        ),
    ),
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
MTF_TREND_EMA_VOL_INTERP_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_MTF_TREND_EMA_VOL_INTERP_ENABLED",
    True,
)
MTF_TREND_EMA_VOL_LOW_PIPS: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_MTF_TREND_EMA_VOL_LOW_PIPS", "2.4"))
)
MTF_TREND_EMA_VOL_HIGH_PIPS: float = max(
    MTF_TREND_EMA_VOL_LOW_PIPS + 0.1,
    float(os.getenv("SCALP_PING_5S_MTF_TREND_EMA_VOL_HIGH_PIPS", "9.0")),
)
MTF_TREND_EMA_FAST_WEIGHT_LOW_VOL: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_MTF_TREND_EMA_FAST_WEIGHT_LOW_VOL",
                "0.0",
            )
        ),
    ),
)
MTF_TREND_EMA_FAST_WEIGHT_HIGH_VOL: float = max(
    MTF_TREND_EMA_FAST_WEIGHT_LOW_VOL,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_MTF_TREND_EMA_FAST_WEIGHT_HIGH_VOL",
                str(MTF_TREND_EMA_FAST_WEIGHT_LOW_VOL),
            )
        ),
    ),
)
MTF_TREND_GAP_SCALE_LOW_VOL: float = max(
    0.1,
    float(os.getenv("SCALP_PING_5S_MTF_TREND_GAP_SCALE_LOW_VOL", "0.65")),
)
MTF_TREND_GAP_SCALE_HIGH_VOL: float = max(
    MTF_TREND_GAP_SCALE_LOW_VOL,
    float(os.getenv("SCALP_PING_5S_MTF_TREND_GAP_SCALE_HIGH_VOL", "0.65")),
)
MTF_TREND_GAP_NORM_MIN_PIPS: float = max(
    0.2,
    float(os.getenv("SCALP_PING_5S_MTF_TREND_GAP_NORM_MIN_PIPS", "0.80")),
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
HORIZON_OPPOSITE_BLOCK_MIN_MULT: float = max(
    0.05,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_HORIZON_OPPOSITE_BLOCK_MIN_MULT",
                "0.15",
            )
        ),
    ),
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

M1_TREND_SCALE_ENABLED: bool = _bool_env("SCALP_PING_5S_M1_TREND_SCALE_ENABLED", True)
M1_TREND_ALIGN_SCORE_MIN: float = max(
    0.05,
    min(
        0.95,
        float(os.getenv("SCALP_PING_5S_M1_TREND_ALIGN_SCORE_MIN", "0.20")),
    ),
)
M1_TREND_VOL_INTERP_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_M1_TREND_VOL_INTERP_ENABLED",
    True,
)
M1_TREND_VOL_LOW_PIPS: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_M1_TREND_VOL_LOW_PIPS", "3.0"))
)
M1_TREND_VOL_HIGH_PIPS: float = max(
    M1_TREND_VOL_LOW_PIPS + 0.1,
    float(os.getenv("SCALP_PING_5S_M1_TREND_VOL_HIGH_PIPS", "10.0")),
)
M1_TREND_ALIGN_SCORE_MIN_LOW_VOL: float = max(
    0.05,
    min(
        0.95,
        float(os.getenv("SCALP_PING_5S_M1_TREND_ALIGN_SCORE_MIN_LOW_VOL", str(M1_TREND_ALIGN_SCORE_MIN))),
    ),
)
M1_TREND_ALIGN_SCORE_MIN_HIGH_VOL: float = max(
    M1_TREND_ALIGN_SCORE_MIN_LOW_VOL,
    min(
        0.95,
        float(
            os.getenv(
                "SCALP_PING_5S_M1_TREND_ALIGN_SCORE_MIN_HIGH_VOL",
                str(M1_TREND_ALIGN_SCORE_MIN),
            )
        ),
    ),
)
M1_TREND_OPPOSITE_SCORE: float = max(
    M1_TREND_ALIGN_SCORE_MIN,
    min(1.0, float(os.getenv("SCALP_PING_5S_M1_TREND_OPPOSITE_SCORE", "0.28"))),
)
M1_TREND_OPPOSITE_SCORE_LOW_VOL: float = max(
    0.05,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_M1_TREND_OPPOSITE_SCORE_LOW_VOL",
                str(M1_TREND_OPPOSITE_SCORE),
            )
        ),
    ),
)
M1_TREND_OPPOSITE_SCORE_HIGH_VOL: float = max(
    M1_TREND_OPPOSITE_SCORE_LOW_VOL,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_M1_TREND_OPPOSITE_SCORE_HIGH_VOL",
                str(M1_TREND_OPPOSITE_SCORE),
            )
        ),
    ),
)
M1_TREND_OPPOSITE_UNITS_MULT: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_M1_TREND_OPPOSITE_UNITS_MULT", "0.72"))),
)
M1_TREND_ALIGN_BOOST_MAX: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_M1_TREND_ALIGN_BOOST_MAX", "0.28"))),
)
M1_TREND_ALIGN_BOOST_MAX_LOW_VOL: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_M1_TREND_ALIGN_BOOST_MAX_LOW_VOL",
                str(M1_TREND_ALIGN_BOOST_MAX),
            )
        ),
    ),
)
M1_TREND_ALIGN_BOOST_MAX_HIGH_VOL: float = max(
    M1_TREND_ALIGN_BOOST_MAX_LOW_VOL,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_M1_TREND_ALIGN_BOOST_MAX_HIGH_VOL",
                str(M1_TREND_ALIGN_BOOST_MAX),
            )
        ),
    ),
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
EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT: float = max(
    0.10,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT",
                (
                    "0.42"
                    if _IS_B_OR_C_PREFIX
                    else "0.68"
                ),
            )
        ),
    ),
)
EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT: float = max(
    0.05,
    min(
        EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT,
        float(
            os.getenv(
                "SCALP_PING_5S_EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT",
                (
                    "0.30"
                    if _IS_B_OR_C_PREFIX
                    else str(EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT)
                ),
            )
        ),
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
EXTREMA_REQUIRE_M1_M5_AGREE_LONG: bool = _bool_env(
    "SCALP_PING_5S_EXTREMA_REQUIRE_M1_M5_AGREE_LONG",
    EXTREMA_REQUIRE_M1_M5_AGREE,
)
EXTREMA_REQUIRE_M1_M5_AGREE_SHORT: bool = _bool_env(
    "SCALP_PING_5S_EXTREMA_REQUIRE_M1_M5_AGREE_SHORT",
    True if _IS_B_OR_C_PREFIX else EXTREMA_REQUIRE_M1_M5_AGREE,
)
EXTREMA_TECH_FILTER_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_EXTREMA_TECH_FILTER_ENABLED",
    False,
)
EXTREMA_TECH_FILTER_ALLOW_BLOCK_TO_SOFT: bool = _bool_env(
    "SCALP_PING_5S_EXTREMA_TECH_FILTER_ALLOW_BLOCK_TO_SOFT",
    True,
)
EXTREMA_TECH_FILTER_ADX_WEAK_MAX: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_EXTREMA_TECH_FILTER_ADX_WEAK_MAX", "22.0")),
)
EXTREMA_TECH_FILTER_LONG_TOP_RSI_MAX: float = max(
    1.0,
    min(99.0, float(os.getenv("SCALP_PING_5S_EXTREMA_TECH_FILTER_LONG_TOP_RSI_MAX", "45.0"))),
)
EXTREMA_TECH_FILTER_SHORT_BOTTOM_RSI_MIN: float = max(
    1.0,
    min(99.0, float(os.getenv("SCALP_PING_5S_EXTREMA_TECH_FILTER_SHORT_BOTTOM_RSI_MIN", "55.0"))),
)
EXTREMA_TECH_FILTER_REQ_EMA_GAP_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_EXTREMA_TECH_FILTER_REQ_EMA_GAP_PIPS", "0.15")),
)
EXTREMA_TECH_FILTER_BLOCK_SOFT_MULT: float = max(
    0.05,
    min(1.0, float(os.getenv("SCALP_PING_5S_EXTREMA_TECH_FILTER_BLOCK_SOFT_MULT", "0.55"))),
)
EXTREMA_SOFT_UNITS_MULT: float = max(
    0.10,
    min(1.0, float(os.getenv("SCALP_PING_5S_EXTREMA_SOFT_UNITS_MULT", "0.68"))),
)
EXTREMA_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_EXTREMA_LOG_INTERVAL_SEC", "8.0")),
)
EXTREMA_REVERSAL_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_EXTREMA_REVERSAL_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT: bool = _bool_env(
    "SCALP_PING_5S_EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT",
    False if _IS_B_OR_C_PREFIX else True,
)
EXTREMA_REVERSAL_MIN_SCORE: float = max(
    0.1,
    float(
        os.getenv(
            "SCALP_PING_5S_EXTREMA_REVERSAL_MIN_SCORE",
            "1.45" if _IS_B_OR_C_PREFIX else "1.80",
        )
    ),
)
EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE: float = max(
    0.1,
    float(
        os.getenv(
            "SCALP_PING_5S_EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE",
            "2.10" if _IS_B_OR_C_PREFIX else str(EXTREMA_REVERSAL_MIN_SCORE),
        )
    ),
)
EXTREMA_REVERSAL_UNITS_MULT: float = max(
    0.1,
    min(
        2.0,
        float(
            os.getenv(
                "SCALP_PING_5S_EXTREMA_REVERSAL_UNITS_MULT",
                "1.00" if _IS_B_OR_C_PREFIX else "0.95",
            )
        ),
    ),
)
EXTREMA_REVERSAL_CONFIDENCE_ADD: int = max(
    0,
    int(float(os.getenv("SCALP_PING_5S_EXTREMA_REVERSAL_CONFIDENCE_ADD", "4"))),
)
EXTREMA_REVERSAL_RSI_CONFIRM: float = max(
    1.0,
    min(99.0, float(os.getenv("SCALP_PING_5S_EXTREMA_REVERSAL_RSI_CONFIRM", "52.0"))),
)
EXTREMA_REVERSAL_EMA_GAP_MIN_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_EXTREMA_REVERSAL_EMA_GAP_MIN_PIPS", "0.10")),
)
EXTREMA_REVERSAL_CONTINUATION_HEAT_MAX: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_EXTREMA_REVERSAL_CONTINUATION_HEAT_MAX", "0.74"))),
)
EXTREMA_REVERSAL_HORIZON_SCORE_MIN: float = max(
    0.0,
    min(1.0, float(os.getenv("SCALP_PING_5S_EXTREMA_REVERSAL_HORIZON_SCORE_MIN", "0.24"))),
)
EXTREMA_REVERSAL_HORIZON_AGREE_MIN: int = max(
    1,
    int(float(os.getenv("SCALP_PING_5S_EXTREMA_REVERSAL_HORIZON_AGREE_MIN", "3"))),
)

TECH_ROUTER_ENABLED: bool = _bool_env("SCALP_PING_5S_TECH_ROUTER_ENABLED", True)
TECH_ROUTER_MTF_BLOCK_UNITS_MULT: float = max(
    0.1,
    min(1.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_MTF_BLOCK_UNITS_MULT", "0.38"))),
)
TECH_ROUTER_HORIZON_BLOCK_UNITS_MULT: float = max(
    0.1,
    min(1.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_HORIZON_BLOCK_UNITS_MULT", "0.42"))),
)
TECH_ROUTER_DIRECTION_BLOCK_UNITS_MULT: float = max(
    0.1,
    min(1.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_DIRECTION_BLOCK_UNITS_MULT", "0.50"))),
)
TECH_ROUTER_LOOKAHEAD_BLOCK_UNITS_MULT: float = max(
    0.1,
    min(1.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_LOOKAHEAD_BLOCK_UNITS_MULT", "0.52"))),
)
TECH_ROUTER_EXTREMA_BLOCK_UNITS_MULT: float = max(
    0.1,
    min(1.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_EXTREMA_BLOCK_UNITS_MULT", "0.45"))),
)
TECH_ROUTER_COUNTER_TP_MULT: float = max(
    0.3,
    min(2.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_COUNTER_TP_MULT", "0.86"))),
)
TECH_ROUTER_COUNTER_SL_MULT: float = max(
    0.3,
    min(2.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_COUNTER_SL_MULT", "0.92"))),
)
TECH_ROUTER_COUNTER_HOLD_MULT: float = max(
    0.2,
    min(2.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_COUNTER_HOLD_MULT", "0.68"))),
)
TECH_ROUTER_COUNTER_HARD_LOSS_MULT: float = max(
    0.2,
    min(2.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_COUNTER_HARD_LOSS_MULT", "0.88"))),
)
TECH_ROUTER_EDGE_TP_BOOST_MAX: float = max(
    0.0,
    min(2.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_EDGE_TP_BOOST_MAX", "0.18"))),
)
TECH_ROUTER_EDGE_HOLD_BOOST_MAX: float = max(
    0.0,
    min(2.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_EDGE_HOLD_BOOST_MAX", "0.25"))),
)
TECH_ROUTER_EDGE_HARD_LOSS_BOOST_MAX: float = max(
    0.0,
    min(2.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_EDGE_HARD_LOSS_BOOST_MAX", "0.15"))),
)
TECH_ROUTER_HOLD_MIN_SEC: float = max(
    5.0,
    float(os.getenv("SCALP_PING_5S_TECH_ROUTER_HOLD_MIN_SEC", "25.0")),
)
TECH_ROUTER_HOLD_MAX_MULT: float = max(
    0.5,
    min(3.0, float(os.getenv("SCALP_PING_5S_TECH_ROUTER_HOLD_MAX_MULT", "1.40"))),
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
#
# Dynamic same-side cap:
# Tighten only when side alignment is weak or adverse stacking is building.
# This preserves high-conviction runs while reducing same-minute cluster risk.
DYNAMIC_DIRECTION_CAP_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
DYNAMIC_DIRECTION_CAP_MIN: int = max(
    1, int(float(os.getenv("SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_MIN", "1")))
)
DYNAMIC_DIRECTION_CAP_WEAK_CAP: int = max(
    DYNAMIC_DIRECTION_CAP_MIN,
    int(float(os.getenv("SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_WEAK_CAP", "2"))),
)
DYNAMIC_DIRECTION_CAP_WEAK_BIAS_SCORE: float = max(
    0.0,
    min(
        1.0,
        float(os.getenv("SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_WEAK_BIAS_SCORE", "0.52")),
    ),
)
DYNAMIC_DIRECTION_CAP_WEAK_HORIZON_SCORE: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_WEAK_HORIZON_SCORE",
                "0.56",
            )
        ),
    ),
)
DYNAMIC_DIRECTION_CAP_ADVERSE_ACTIVE_START: int = max(
    1,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_ADVERSE_ACTIVE_START",
                "2",
            )
        )
    ),
)
DYNAMIC_DIRECTION_CAP_ADVERSE_DD_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_ADVERSE_DD_PIPS", "0.45")),
)
DYNAMIC_DIRECTION_CAP_ADVERSE_CAP: int = max(
    DYNAMIC_DIRECTION_CAP_MIN,
    int(float(os.getenv("SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_ADVERSE_CAP", "2"))),
)
DYNAMIC_DIRECTION_CAP_METRICS_ADVERSE_CAP: int = max(
    DYNAMIC_DIRECTION_CAP_MIN,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_METRICS_ADVERSE_CAP",
                "1",
            )
        )
    ),
)
DYNAMIC_DIRECTION_CAP_LOG_INTERVAL_SEC: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_LOG_INTERVAL_SEC", "8.0")),
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

MIN_UNITS: int = max(1, int(float(os.getenv("SCALP_PING_5S_MIN_UNITS", "2000"))))
MAX_UNITS: int = max(MIN_UNITS, int(float(os.getenv("SCALP_PING_5S_MAX_UNITS", "25000"))))
BASE_ENTRY_UNITS: int = max(
    MIN_UNITS, int(float(os.getenv("SCALP_PING_5S_BASE_ENTRY_UNITS", "10000")))
)

CONFIDENCE_FLOOR: int = max(0, int(float(os.getenv("SCALP_PING_5S_CONF_FLOOR", "58"))))
CONFIDENCE_CEIL: int = max(
    CONFIDENCE_FLOOR + 1, int(float(os.getenv("SCALP_PING_5S_CONF_CEIL", "92")))
)
CONFIDENCE_SCALE_MIN_MULT: float = max(
    0.10,
    min(
        2.0,
        float(
            os.getenv(
                "SCALP_PING_5S_CONF_SCALE_MIN_MULT",
                "0.72" if _IS_B_OR_C_PREFIX else "0.65",
            )
        ),
    ),
)
CONFIDENCE_SCALE_MAX_MULT: float = max(
    CONFIDENCE_SCALE_MIN_MULT,
    min(
        2.5,
        float(
            os.getenv(
                "SCALP_PING_5S_CONF_SCALE_MAX_MULT",
                "1.00" if _IS_B_OR_C_PREFIX else "1.15",
            )
        ),
    ),
)
ENTRY_PROBABILITY_ALIGN_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
ENTRY_PROBABILITY_ALIGN_DIRECTION_WEIGHT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_DIRECTION_WEIGHT", "0.15")),
)
ENTRY_PROBABILITY_ALIGN_HORIZON_WEIGHT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_HORIZON_WEIGHT", "0.65")),
)
ENTRY_PROBABILITY_ALIGN_M1_WEIGHT: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_M1_WEIGHT", "0.20")),
)
ENTRY_PROBABILITY_ALIGN_BOOST_MAX: float = max(
    0.0,
    min(
        0.50,
        float(os.getenv("SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_BOOST_MAX", "0.08")),
    ),
)
ENTRY_PROBABILITY_ALIGN_PENALTY_MAX: float = max(
    0.0,
    min(
        0.95,
        float(os.getenv("SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_PENALTY_MAX", "0.45")),
    ),
)
ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX: float = max(
    0.0,
    min(
        0.95,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX",
                "0.25",
            )
        ),
    ),
)
ENTRY_PROBABILITY_ALIGN_REVERT_PENALTY_MULT: float = max(
    0.10,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_REVERT_PENALTY_MULT",
                "0.82",
            )
        ),
    ),
)
ENTRY_PROBABILITY_ALIGN_MIN: float = max(
    0.0,
    min(
        1.0,
        float(os.getenv("SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_MIN", "0.0")),
    ),
)
ENTRY_PROBABILITY_ALIGN_MAX: float = max(
    ENTRY_PROBABILITY_ALIGN_MIN,
    min(
        1.0,
        float(os.getenv("SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_MAX", "1.0")),
    ),
)
ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN",
                "0.70",
            )
        ),
    ),
)
ENTRY_PROBABILITY_ALIGN_FLOOR: float = max(
    0.0,
    min(
        1.0,
        float(os.getenv("SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_FLOOR", "0.46")),
    ),
)
ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT: bool = _bool_env(
    "SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT",
    True if _IS_B_OR_C_PREFIX else False,
)
ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER",
                "0.30" if _IS_B_OR_C_PREFIX else "1.00",
            )
        ),
    ),
)
ENTRY_PROBABILITY_ALIGN_UNITS_FOLLOW_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_UNITS_FOLLOW_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT: float = max(
    0.10,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT",
                "0.55",
            )
        ),
    ),
)
ENTRY_PROBABILITY_ALIGN_UNITS_MAX_MULT: float = max(
    ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT,
    min(
        2.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_ALIGN_UNITS_MAX_MULT",
                "1.00",
            )
        ),
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_ENABLED",
    True if _IS_B_OR_C_PREFIX else False,
)
ENTRY_PROBABILITY_BAND_ALLOC_LOOKBACK_TRADES: int = max(
    20,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_LOOKBACK_TRADES",
                "180",
            )
        )
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_CACHE_TTL_SEC: float = max(
    0.2,
    float(
        os.getenv(
            "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_CACHE_TTL_SEC",
            "4.0",
        )
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_LOW_THRESHOLD: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_LOW_THRESHOLD",
                "0.70",
            )
        ),
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_HIGH_THRESHOLD: float = max(
    ENTRY_PROBABILITY_BAND_ALLOC_LOW_THRESHOLD + 0.01,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_HIGH_THRESHOLD",
                "0.90",
            )
        ),
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_MIN_TRADES_PER_BAND: int = max(
    5,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_MIN_TRADES_PER_BAND",
                "20",
            )
        )
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_HIGH_REDUCE_MAX: float = max(
    0.0,
    min(
        0.90,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_HIGH_REDUCE_MAX",
                "0.45",
            )
        ),
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_LOW_BOOST_MAX: float = max(
    0.0,
    min(
        0.90,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_LOW_BOOST_MAX",
                "0.30",
            )
        ),
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_GAP_PIPS_REF: float = max(
    0.05,
    float(
        os.getenv("SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_GAP_PIPS_REF", "1.0")
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_GAP_WIN_RATE_REF: float = max(
    0.01,
    float(
        os.getenv(
            "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_GAP_WIN_RATE_REF",
            "0.20",
        )
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_GAP_SL_RATE_REF: float = max(
    0.01,
    float(
        os.getenv(
            "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_GAP_SL_RATE_REF",
            "0.20",
        )
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_SAMPLE_STRONG_TRADES: int = max(
    5,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_SAMPLE_STRONG_TRADES",
                "60",
            )
        )
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT: float = max(
    0.10,
    min(
        2.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT",
                "0.55",
            )
        ),
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MAX_MULT: float = max(
    ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT,
    min(
        3.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MAX_MULT",
                "1.35",
            )
        ),
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_ENABLED",
    True,
)
ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_LOOKBACK_TRADES: int = max(
    20,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_LOOKBACK_TRADES",
                "120",
            )
        )
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_CACHE_TTL_SEC: float = max(
    0.2,
    float(
        os.getenv(
            "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_CACHE_TTL_SEC",
            "4.0",
        )
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_GAIN: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_GAIN",
            "0.30",
        )
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT: float = max(
    0.10,
    min(
        2.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT",
                "0.85",
            )
        ),
    ),
)
ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT: float = max(
    ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT,
    min(
        3.0,
        float(
            os.getenv(
                "SCALP_PING_5S_ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT",
                "1.08",
            )
        ),
    ),
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

TP_BASE_PIPS: float = max(0.2, float(os.getenv("SCALP_PING_5S_TP_BASE_PIPS", "0.30")))
TP_NET_MIN_PIPS: float = max(0.1, float(os.getenv("SCALP_PING_5S_TP_NET_MIN_PIPS", "0.30")))
TP_MAX_PIPS: float = max(TP_BASE_PIPS, float(os.getenv("SCALP_PING_5S_TP_MAX_PIPS", "0.80")))
TP_ENABLED: bool = _bool_env("SCALP_PING_5S_TP_ENABLED", True)
TP_MOMENTUM_BONUS_MAX: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_TP_MOMENTUM_BONUS_MAX", "0.2"))
)
SHORT_TP_BASE_PIPS: float = max(
    0.2, float(os.getenv("SCALP_PING_5S_SHORT_TP_BASE_PIPS", str(TP_BASE_PIPS)))
)
SHORT_TP_NET_MIN_PIPS: float = max(
    0.1,
    float(os.getenv("SCALP_PING_5S_SHORT_TP_NET_MIN_PIPS", str(TP_NET_MIN_PIPS))),
)
SHORT_TP_MAX_PIPS: float = max(
    SHORT_TP_BASE_PIPS,
    float(os.getenv("SCALP_PING_5S_SHORT_TP_MAX_PIPS", str(TP_MAX_PIPS))),
)
SHORT_TP_MOMENTUM_BONUS_MAX: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_SHORT_TP_MOMENTUM_BONUS_MAX", str(TP_MOMENTUM_BONUS_MAX)))
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
    min(1.6, float(os.getenv("SCALP_PING_5S_TP_TIME_MULT_MAX", "1.0"))),
)
TP_VOL_ADAPT_ENABLED: bool = _bool_env("SCALP_PING_5S_TP_VOL_ADAPT_ENABLED", True)
TP_VOL_LOW_PIPS: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_TP_VOL_LOW_PIPS", "3.0"))
)
TP_VOL_HIGH_PIPS: float = max(
    TP_VOL_LOW_PIPS + 0.1,
    float(os.getenv("SCALP_PING_5S_TP_VOL_HIGH_PIPS", "10.0")),
)
TP_VOL_MULT_LOW_VOL_MIN: float = max(
    0.1,
    min(1.2, float(os.getenv("SCALP_PING_5S_TP_VOL_MULT_LOW_VOL_MIN", "0.90"))),
)
TP_VOL_MULT_LOW_VOL_MAX: float = max(
    TP_VOL_MULT_LOW_VOL_MIN,
    min(1.8, float(os.getenv("SCALP_PING_5S_TP_VOL_MULT_LOW_VOL_MAX", "1.00"))),
)
TP_VOL_MULT_HIGH_VOL_MIN: float = max(
    0.1,
    min(1.8, float(os.getenv("SCALP_PING_5S_TP_VOL_MULT_HIGH_VOL_MIN", "1.00"))),
)
TP_VOL_MULT_HIGH_VOL_MAX: float = max(
    TP_VOL_MULT_HIGH_VOL_MIN,
    min(2.5, float(os.getenv("SCALP_PING_5S_TP_VOL_MULT_HIGH_VOL_MAX", "1.20"))),
)
TP_VOL_EXTEND_MAX_MULT: float = max(
    1.0,
    float(os.getenv("SCALP_PING_5S_TP_VOL_EXTEND_MAX_MULT", "1.35")),
)
TP_NET_MIN_FLOOR_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_TP_NET_MIN_FLOOR_PIPS", "0.10"))
)
ENTRY_CHASE_MAX_PIPS: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_ENTRY_CHASE_MAX_PIPS", "1.4"))
)

SL_BASE_PIPS: float = max(0.2, float(os.getenv("SCALP_PING_5S_SL_BASE_PIPS", "1.40")))
SL_MIN_PIPS: float = max(0.2, float(os.getenv("SCALP_PING_5S_SL_MIN_PIPS", "1.00")))
SL_MAX_PIPS: float = max(SL_MIN_PIPS, float(os.getenv("SCALP_PING_5S_SL_MAX_PIPS", "4.20")))
SL_SPREAD_MULT: float = max(0.0, float(os.getenv("SCALP_PING_5S_SL_SPREAD_MULT", "1.2")))
SL_SPREAD_BUFFER_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_SL_SPREAD_BUFFER_PIPS", "0.30"))
)
SL_VOL_SHRINK_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_SL_VOL_SHRINK_ENABLED",
    True,
)
SL_VOL_SHRINK_MIN_MULT: float = max(
    0.2,
    min(1.0, float(os.getenv("SCALP_PING_5S_SL_VOL_SHRINK_MIN_MULT", "0.70"))),
)
SHORT_SL_BASE_PIPS: float = max(
    0.2, float(os.getenv("SCALP_PING_5S_SHORT_SL_BASE_PIPS", str(SL_BASE_PIPS)))
)
SHORT_SL_MIN_PIPS: float = max(
    0.2, float(os.getenv("SCALP_PING_5S_SHORT_SL_MIN_PIPS", str(SL_MIN_PIPS)))
)
SHORT_SL_MAX_PIPS: float = max(
    SHORT_SL_MIN_PIPS, float(os.getenv("SCALP_PING_5S_SHORT_SL_MAX_PIPS", str(SL_MAX_PIPS)))
)

SNAPSHOT_FALLBACK_ENABLED: bool = _bool_env("SCALP_PING_5S_SNAPSHOT_FALLBACK_ENABLED", True)
SNAPSHOT_MIN_INTERVAL_SEC: float = max(
    0.1, float(os.getenv("SCALP_PING_5S_SNAPSHOT_MIN_INTERVAL_SEC", "0.2"))
)
SNAPSHOT_FETCH_TIMEOUT_SEC: float = max(
    0.5, float(os.getenv("SCALP_PING_5S_SNAPSHOT_FETCH_TIMEOUT_SEC", "3.0"))
)
SNAPSHOT_FETCH_RETRY_BASE_SEC: float = max(
    0.2, float(os.getenv("SCALP_PING_5S_SNAPSHOT_FETCH_RETRY_BASE_SEC", "0.75"))
)
SNAPSHOT_FETCH_RETRY_MAX_SEC: float = max(
    SNAPSHOT_FETCH_RETRY_BASE_SEC,
    float(os.getenv("SCALP_PING_5S_SNAPSHOT_FETCH_RETRY_MAX_SEC", "8.0")),
)
SNAPSHOT_FETCH_RETRY_BACKOFF_MULTIPLIER: float = max(
    1.1, float(os.getenv("SCALP_PING_5S_SNAPSHOT_FETCH_RETRY_BACKOFF_MULTIPLIER", "1.8"))
)
SNAPSHOT_FETCH_FAILURE_STALE_MODE_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_SNAPSHOT_FAILURE_STALE_MODE_ENABLED", True
)
SNAPSHOT_FETCH_FAILURE_STALE_THRESHOLD: int = max(
    1,
    int(float(os.getenv("SCALP_PING_5S_SNAPSHOT_FAILURE_STALE_THRESHOLD", "3"))),
)
SNAPSHOT_FETCH_FAILURE_STALE_MAX_AGE_MS: float = max(
    MAX_TICK_AGE_MS,
    float(os.getenv("SCALP_PING_5S_SNAPSHOT_FAILURE_STALE_MAX_AGE_MS", "3200.0")),
)
SNAPSHOT_FETCH_FAILURE_STALE_UNITS_SCALE: float = max(
    0.1,
    min(1.0, float(os.getenv("SCALP_PING_5S_SNAPSHOT_FAILURE_STALE_UNITS_SCALE", "0.55"))),
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

FORCE_EXIT_ENABLED: bool = _bool_env("SCALP_PING_5S_FORCE_EXIT_ENABLED", True)
FORCE_EXIT_MAX_HOLD_SEC: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_HOLD_SEC", "150"))
)
FORCE_EXIT_MAX_FLOATING_LOSS_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS", "3.5"))
)
FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC", "0.0"))
)
FORCE_EXIT_FLOATING_LOSS_MIN_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_FLOATING_LOSS_MIN_PIPS", "0.0"))
)
SHORT_FORCE_EXIT_MAX_HOLD_SEC: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_SHORT_FORCE_EXIT_MAX_HOLD_SEC",
            str(FORCE_EXIT_MAX_HOLD_SEC),
        )
    ),
)
SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS",
            str(FORCE_EXIT_MAX_FLOATING_LOSS_PIPS),
        )
    ),
)
FORCE_EXIT_RECOVERY_WINDOW_SEC: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_RECOVERY_WINDOW_SEC", "0.0"))
)
FORCE_EXIT_RECOVERABLE_LOSS_PIPS: float = max(
    0.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_RECOVERABLE_LOSS_PIPS", "0.0"))
)
FORCE_EXIT_VOL_ADAPT_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_FORCE_EXIT_VOL_ADAPT_ENABLED",
    True,
)
FORCE_EXIT_VOL_HOLD_MIN_MULT: float = max(
    0.2,
    min(1.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_VOL_HOLD_MIN_MULT", "0.55"))),
)
FORCE_EXIT_VOL_LOSS_MIN_MULT: float = max(
    0.2,
    min(1.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_VOL_LOSS_MIN_MULT", "0.50"))),
)
FORCE_EXIT_VOL_HOLD_MAX_MULT: float = max(
    FORCE_EXIT_VOL_HOLD_MIN_MULT,
    min(1.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_VOL_HOLD_MAX_MULT", "1.0"))),
)
FORCE_EXIT_VOL_LOSS_MAX_MULT: float = max(
    FORCE_EXIT_VOL_LOSS_MIN_MULT,
    min(1.0, float(os.getenv("SCALP_PING_5S_FORCE_EXIT_VOL_LOSS_MAX_MULT", "1.0"))),
)
FORCE_EXIT_BID_ASK_BUFFER_PIPS: float = max(
    0.0,
    float(
        os.getenv(
            "SCALP_PING_5S_FORCE_EXIT_BID_ASK_BUFFER_PIPS",
            "0.06",
        )
    ),
)
_FORCE_EXIT_DEFAULT_MAX_ACTIONS = "2" if _IS_B_OR_C_PREFIX else "0"
FORCE_EXIT_MAX_ACTIONS: int = max(
    0,
    int(
        float(
            os.getenv(
                "SCALP_PING_5S_FORCE_EXIT_MAX_ACTIONS",
                _FORCE_EXIT_DEFAULT_MAX_ACTIONS,
            )
        )
    ),
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

FORCE_EXIT_MOMENTUM_STALL_ENABLED: bool = _bool_env(
    "SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_ENABLED",
    True,
)
FORCE_EXIT_MOMENTUM_STALL_WINDOW_SEC: float = max(
    0.4,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_WINDOW_SEC", "2.0")),
)
FORCE_EXIT_MOMENTUM_STALL_MIN_TICKS: int = max(
    3,
    int(float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_MIN_TICKS", "6"))),
)
FORCE_EXIT_MOMENTUM_STALL_MIN_EARLY_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_MIN_EARLY_PIPS", "0.25")),
)
FORCE_EXIT_MOMENTUM_STALL_MIN_LATE_PIPS: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_MIN_LATE_PIPS", "0.20")),
)
FORCE_EXIT_MOMENTUM_STALL_MIN_HOLD_SEC: float = max(
    0.0,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_MIN_HOLD_SEC", "1.0")),
)
FORCE_EXIT_MOMENTUM_STALL_FLAT_REMAIN_RATIO: float = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv(
                "SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_FLAT_REMAIN_RATIO",
                "0.42",
            )
        ),
    ),
)
FORCE_EXIT_MOMENTUM_STALL_MAX_TICK_AGE_MS: float = max(
    150.0,
    float(os.getenv("SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_MAX_TICK_AGE_MS", "1200")),
)
FORCE_EXIT_MOMENTUM_STALL_TP_REASON: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_TP_REASON", "TAKE_PROFIT_ORDER").strip()
    or "TAKE_PROFIT_ORDER"
)
FORCE_EXIT_MOMENTUM_STALL_LOSS_REASON: str = (
    os.getenv("SCALP_PING_5S_FORCE_EXIT_MOMENTUM_STALL_LOSS_REASON", "momentum_stop_loss").strip()
    or "momentum_stop_loss"
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

# Backward-compatible names kept for older call sites.
FORCE_EXIT_REASON_TIME: str = FORCE_EXIT_REASON
FORCE_EXIT_REASON_MAX_FLOATING_LOSS: str = FORCE_EXIT_MAX_FLOATING_LOSS_REASON
FORCE_EXIT_REASON_RECOVERY: str = FORCE_EXIT_RECOVERY_REASON
FORCE_EXIT_REASON_GIVEBACK: str = FORCE_EXIT_GIVEBACK_REASON

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
_IS_B_OR_C_VARIANT: bool = STRATEGY_TAG.startswith(("scalp_ping_5s_b", "scalp_ping_5s_c", "scalp_ping_5s_d"))
USE_SL: bool = _bool_env("SCALP_PING_5S_USE_SL", _IS_B_OR_C_VARIANT)
DISABLE_ENTRY_HARD_STOP: bool = _bool_env(
    "SCALP_PING_5S_DISABLE_ENTRY_HARD_STOP", not USE_SL
)
