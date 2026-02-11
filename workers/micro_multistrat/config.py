from __future__ import annotations

import os

ENV_PREFIX = "MICRO_MULTI"

def _parse_hours(raw: str) -> set[int]:
    hours: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            try:
                start = int(float(start_s))
                end = int(float(end_s))
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for h in range(start, end + 1):
                if 0 <= h <= 23:
                    hours.add(h)
            continue
        try:
            h = int(float(token))
        except ValueError:
            continue
        if 0 <= h <= 23:
            hours.add(h)
    return hours


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}

POCKET = "micro"
LOOP_INTERVAL_SEC = float(os.getenv("MICRO_MULTI_LOOP_INTERVAL_SEC", "8.0"))
ENABLED = os.getenv("MICRO_MULTI_ENABLED", "0").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[MicroMulti]"

CONFIDENCE_FLOOR = 35
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("MICRO_MULTI_MIN_UNITS", "1500"))
BASE_ENTRY_UNITS = int(os.getenv("MICRO_MULTI_BASE_UNITS", "20000"))
MAX_MARGIN_USAGE = float(os.getenv("MICRO_MULTI_MAX_MARGIN_USAGE", "0.9"))

CAP_MIN = float(os.getenv("MICRO_MULTI_CAP_MIN", "0.15"))
CAP_MAX = float(os.getenv("MICRO_MULTI_CAP_MAX", "0.95"))

# TrendMomentumMicro が沈んだ時間帯を任意でブロック（デフォルト無効）
TREND_BLOCK_HOURS_UTC = frozenset(
    _parse_hours(os.getenv("MICRO_TREND_BLOCK_HOURS_UTC", ""))
)

MAX_FACTOR_AGE_SEC = float(os.getenv("MICRO_MULTI_MAX_FACTOR_AGE_SEC", "90.0"))

# Trend/Projection reconciliation (avoid counter-trend traps without reducing frequency).
TREND_FLIP_ENABLED = _bool("MICRO_MULTI_TREND_FLIP_ENABLED", True)
TREND_FLIP_GAP_PIPS = float(os.getenv("MICRO_MULTI_TREND_FLIP_GAP_PIPS", "0.6"))
TREND_FLIP_ADX_MIN = float(os.getenv("MICRO_MULTI_TREND_FLIP_ADX_MIN", "20.0"))
TREND_FLIP_TP_MULT = float(os.getenv("MICRO_MULTI_TREND_FLIP_TP_MULT", "1.12"))
TREND_FLIP_SL_MULT = float(os.getenv("MICRO_MULTI_TREND_FLIP_SL_MULT", "0.95"))

PROJ_FLIP_ENABLED = _bool("MICRO_MULTI_PROJ_FLIP_ENABLED", True)
PROJ_CONFLICT_ALLOW = _bool("MICRO_MULTI_PROJ_CONFLICT_ALLOW", True)

# Fresh factor fallback (avoid stale M1 driving wrong direction).
FRESH_TICKS_ON_STALE = _bool("MICRO_MULTI_FRESH_TICKS_ON_STALE", True)
FRESH_TICKS_LOOKBACK_SEC = float(os.getenv("MICRO_MULTI_FRESH_TICKS_LOOKBACK_SEC", "1800.0"))
FRESH_TICKS_MIN_CANDLES = int(os.getenv("MICRO_MULTI_FRESH_TICKS_MIN_CANDLES", "30"))
FRESH_TICKS_REFRESH_SEC = float(os.getenv("MICRO_MULTI_FRESH_TICKS_REFRESH_SEC", "20.0"))

# Range-mode selection bias.
RANGE_ONLY_SCORE = float(os.getenv("MICRO_MULTI_RANGE_ONLY_SCORE", "0.45"))
RANGE_BIAS_SCORE = float(os.getenv("MICRO_MULTI_RANGE_BIAS_SCORE", "0.30"))
RANGE_STRATEGY_BONUS = float(os.getenv("MICRO_MULTI_RANGE_STRATEGY_BONUS", "12"))
RANGE_TREND_PENALTY = float(os.getenv("MICRO_MULTI_RANGE_TREND_PENALTY", "10"))

# Strategy diversity: promote idle strategies without inflating risk sizing.
DIVERSITY_ENABLED = os.getenv("MICRO_MULTI_DIVERSITY_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
DIVERSITY_IDLE_SEC = float(os.getenv("MICRO_MULTI_DIVERSITY_IDLE_SEC", "300"))
DIVERSITY_SCALE_SEC = float(os.getenv("MICRO_MULTI_DIVERSITY_SCALE_SEC", "900"))
DIVERSITY_MAX_BONUS = float(os.getenv("MICRO_MULTI_DIVERSITY_MAX_BONUS", "12"))

# Multi-signal dispatch (variety): send top-N signals per cycle with smaller sizing.
MAX_SIGNALS_PER_CYCLE = int(os.getenv("MICRO_MULTI_MAX_SIGNALS_PER_CYCLE", "2"))
MULTI_SIGNAL_MIN_SCALE = float(os.getenv("MICRO_MULTI_MULTI_SIGNAL_MIN_SCALE", "0.6"))

# Dynamic winner routing from config/dynamic_alloc.json
DYN_ALLOC_ENABLED = os.getenv("MICRO_MULTI_DYN_ALLOC_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
DYN_ALLOC_PATH = os.getenv("MICRO_MULTI_DYN_ALLOC_PATH", "config/dynamic_alloc.json")
DYN_ALLOC_TTL_SEC = float(os.getenv("MICRO_MULTI_DYN_ALLOC_TTL_SEC", "20"))
DYN_ALLOC_MIN_TRADES = int(os.getenv("MICRO_MULTI_DYN_ALLOC_MIN_TRADES", "10"))
DYN_ALLOC_WINNER_SCORE = float(os.getenv("MICRO_MULTI_DYN_ALLOC_WINNER_SCORE", "0.62"))
DYN_ALLOC_LOSER_SCORE = float(os.getenv("MICRO_MULTI_DYN_ALLOC_LOSER_SCORE", "0.28"))
DYN_ALLOC_WINNER_ONLY = os.getenv("MICRO_MULTI_DYN_ALLOC_WINNER_ONLY", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
DYN_ALLOC_LOSER_BLOCK = os.getenv("MICRO_MULTI_DYN_ALLOC_LOSER_BLOCK", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
DYN_ALLOC_SCORE_BONUS = float(os.getenv("MICRO_MULTI_DYN_ALLOC_SCORE_BONUS", "10.0"))
DYN_ALLOC_MULT_MIN = float(os.getenv("MICRO_MULTI_DYN_ALLOC_MULT_MIN", "0.7"))
DYN_ALLOC_MULT_MAX = float(os.getenv("MICRO_MULTI_DYN_ALLOC_MULT_MAX", "1.8"))
