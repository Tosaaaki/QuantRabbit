from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


ENV_PREFIX = "MACDRSIDIV"
PIP_VALUE = 0.01
POCKET = os.getenv("MACDRSIDIV_POCKET", "scalp").strip() or "scalp"
STRATEGY_TAG = (
    os.getenv("MACDRSIDIV_STRATEGY_TAG", "scalp_macd_rsi_div_live").strip()
    or "scalp_macd_rsi_div_live"
)
LOG_PREFIX = os.getenv("MACDRSIDIV_LOG_PREFIX", "[MACD_RSI_DIV]")

ENABLED = _env_bool("MACDRSIDIV_ENABLED", True)
PATTERN_GATE_OPT_IN = _env_bool("MACDRSIDIV_PATTERN_GATE_OPT_IN", True)

LOOP_INTERVAL_SEC = float(os.getenv("MACDRSIDIV_LOOP_INTERVAL_SEC", "2.0"))
COOLDOWN_SEC = float(os.getenv("MACDRSIDIV_COOLDOWN_SEC", "75.0"))
MAX_SPREAD_PIPS = float(os.getenv("MACDRSIDIV_MAX_SPREAD_PIPS", "1.1"))

MAX_ADX = float(os.getenv("MACDRSIDIV_MAX_ADX", "22.0"))
REQUIRE_RANGE_ACTIVE = _env_bool("MACDRSIDIV_REQUIRE_RANGE_ACTIVE", True)
RANGE_MIN_SCORE = float(os.getenv("MACDRSIDIV_RANGE_MIN_SCORE", "0.35"))

RSI_LONG_ARM = float(os.getenv("MACDRSIDIV_RSI_LONG_ARM", "25.0"))
RSI_SHORT_ARM = float(os.getenv("MACDRSIDIV_RSI_SHORT_ARM", "75.0"))
RSI_LONG_ENTRY = float(os.getenv("MACDRSIDIV_RSI_LONG_ENTRY", "30.0"))
RSI_SHORT_ENTRY = float(os.getenv("MACDRSIDIV_RSI_SHORT_ENTRY", "70.0"))
RSI_ARM_TTL_SEC = float(os.getenv("MACDRSIDIV_RSI_ARM_TTL_SEC", "420.0"))

MIN_DIV_SCORE = float(os.getenv("MACDRSIDIV_MIN_DIV_SCORE", "0.15"))
MIN_DIV_STRENGTH = float(os.getenv("MACDRSIDIV_MIN_DIV_STRENGTH", "0.22"))
MAX_DIV_AGE_BARS = int(float(os.getenv("MACDRSIDIV_MAX_DIV_AGE_BARS", "8")))
ALLOW_HIDDEN_DIVERGENCE = _env_bool("MACDRSIDIV_ALLOW_HIDDEN_DIVERGENCE", False)

CONFIDENCE_FLOOR = int(float(os.getenv("MACDRSIDIV_CONFIDENCE_FLOOR", "52")))
CONFIDENCE_CEIL = int(float(os.getenv("MACDRSIDIV_CONFIDENCE_CEIL", "90")))
MIN_UNITS = int(float(os.getenv("MACDRSIDIV_MIN_UNITS", "1200")))
BASE_ENTRY_UNITS = int(float(os.getenv("MACDRSIDIV_BASE_ENTRY_UNITS", "7000")))

CAP_MIN = float(os.getenv("MACDRSIDIV_CAP_MIN", "0.10"))
CAP_MAX = float(os.getenv("MACDRSIDIV_CAP_MAX", "0.85"))

MAX_OPEN_TRADES = int(float(os.getenv("MACDRSIDIV_MAX_OPEN_TRADES", "1")))
MAX_OPEN_TRADES_GLOBAL = int(float(os.getenv("MACDRSIDIV_MAX_OPEN_TRADES_GLOBAL", "0")))
OPEN_TRADES_SCOPE = (os.getenv("MACDRSIDIV_OPEN_TRADES_SCOPE", "tag") or "tag").strip().lower()

SL_ATR_MULT = float(os.getenv("MACDRSIDIV_SL_ATR_MULT", "0.85"))
TP_ATR_MULT = float(os.getenv("MACDRSIDIV_TP_ATR_MULT", "1.10"))
MIN_SL_PIPS = float(os.getenv("MACDRSIDIV_MIN_SL_PIPS", "0.7"))
MAX_SL_PIPS = float(os.getenv("MACDRSIDIV_MAX_SL_PIPS", "3.5"))
MIN_TP_PIPS = float(os.getenv("MACDRSIDIV_MIN_TP_PIPS", "0.8"))
MAX_TP_PIPS = float(os.getenv("MACDRSIDIV_MAX_TP_PIPS", "4.2"))
MIN_TP_RR = float(os.getenv("MACDRSIDIV_MIN_TP_RR", "0.95"))

MIN_FREE_MARGIN_RATIO_HARD = float(os.getenv("MACDRSIDIV_MIN_FREE_MARGIN_RATIO_HARD", "0.08"))
MARGIN_USAGE_HARD = float(os.getenv("MACDRSIDIV_MARGIN_USAGE_HARD", "0.88"))

DYN_ALLOC_ENABLED = _env_bool("MACDRSIDIV_DYN_ALLOC_ENABLED", True)
DYN_ALLOC_PATH = os.getenv("MACDRSIDIV_DYN_ALLOC_PATH", "config/dynamic_alloc.json")
DYN_ALLOC_TTL_SEC = float(os.getenv("MACDRSIDIV_DYN_ALLOC_TTL_SEC", "20"))
DYN_ALLOC_MIN_TRADES = int(float(os.getenv("MACDRSIDIV_DYN_ALLOC_MIN_TRADES", "8")))
DYN_ALLOC_LOSER_SCORE = float(os.getenv("MACDRSIDIV_DYN_ALLOC_LOSER_SCORE", "0.30"))
DYN_ALLOC_LOSER_BLOCK = _env_bool("MACDRSIDIV_DYN_ALLOC_LOSER_BLOCK", True)
DYN_ALLOC_MULT_MIN = float(os.getenv("MACDRSIDIV_DYN_ALLOC_MULT_MIN", "0.7"))
DYN_ALLOC_MULT_MAX = float(os.getenv("MACDRSIDIV_DYN_ALLOC_MULT_MAX", "1.8"))
