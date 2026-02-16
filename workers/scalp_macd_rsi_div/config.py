from __future__ import annotations

import os

ENV_PREFIX = "SCALP_PRECISION"

def _sync_legacy_prefix(prefix: str) -> None:
    legacy_prefix = f"{prefix}_"
    canonical_prefix = f"{ENV_PREFIX}_"
    if not legacy_prefix or not canonical_prefix:
        return
    for key, value in list(os.environ.items()):
        key = str(key)
        if not key.startswith(legacy_prefix):
            continue
        suffix = key[len(legacy_prefix) :]
        if not suffix:
            continue
        canonical_key = f"{canonical_prefix}{suffix}"
        if canonical_key not in os.environ:
            os.environ[canonical_key] = str(value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}

# Backward compatibility:
#  - MACDRSIDIV_* is still used by historical unit files.
#  - SCALP_MACD_RSI_DIV_B_* is used by the micro(B) wrapper.
_sync_legacy_prefix("MACDRSIDIV")
_sync_legacy_prefix("SCALP_MACD_RSI_DIV_B")
_sync_legacy_prefix("SCALP_MACD_RSI_DIV")

POCKET = "scalp"
ENABLED = _env_bool("SCALP_PRECISION_ENABLED", False)
LOOP_INTERVAL_SEC = float(os.getenv("SCALP_PRECISION_LOOP_INTERVAL_SEC", "4.0"))
LOG_PREFIX = os.getenv("SCALP_PRECISION_LOG_PREFIX", "[ScalpPrecision]")

CONFIDENCE_FLOOR = int(float(os.getenv("SCALP_PRECISION_CONF_FLOOR", "32")))
CONFIDENCE_CEIL = int(float(os.getenv("SCALP_PRECISION_CONF_CEIL", "92")))
MIN_ENTRY_CONF = int(float(os.getenv("SCALP_PRECISION_MIN_ENTRY_CONF", str(CONFIDENCE_FLOOR))))

MIN_UNITS = int(float(os.getenv("SCALP_PRECISION_MIN_UNITS", "1000")))
BASE_ENTRY_UNITS = int(float(os.getenv("SCALP_PRECISION_BASE_UNITS", "9000")))
MAX_MARGIN_USAGE = float(os.getenv("SCALP_PRECISION_MAX_MARGIN_USAGE", "0.92"))

CAP_MIN = float(os.getenv("SCALP_PRECISION_CAP_MIN", "0.12"))
CAP_MAX = float(os.getenv("SCALP_PRECISION_CAP_MAX", "0.95"))

COOLDOWN_SEC = float(os.getenv("SCALP_PRECISION_COOLDOWN_SEC", "45"))
MAX_OPEN_TRADES = int(float(os.getenv("SCALP_PRECISION_MAX_OPEN_TRADES", "2")))
MAX_OPEN_TRADES_GLOBAL = int(float(os.getenv("SCALP_PRECISION_MAX_OPEN_TRADES_GLOBAL", "0")))
OPEN_TRADES_SCOPE = os.getenv("SCALP_PRECISION_OPEN_TRADES_SCOPE", "tag").strip().lower()

MAX_SPREAD_PIPS = float(os.getenv("SCALP_PRECISION_MAX_SPREAD_PIPS", "1.2"))

MAX_SIGNALS_PER_CYCLE = int(float(os.getenv("SCALP_PRECISION_MAX_SIGNALS_PER_CYCLE", "1")))

MODE = os.getenv("SCALP_PRECISION_MODE", "spread_revert").strip().lower()
# /home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env may define a global allowlist shared by
# multiple units. Prefer an explicit
# per-unit override when present so a single-strategy unit can run a new mode without editing
# the global env file.
ALLOWLIST_RAW = os.getenv("SCALP_PRECISION_UNIT_ALLOWLIST", os.getenv("SCALP_PRECISION_ALLOWLIST", "")).strip()
MODE_FILTER_ALLOWLIST = _env_bool("SCALP_PRECISION_MODE_FILTER_ALLOWLIST", False)
GUARD_BYPASS_MODES = {
    s.strip().lower()
    for s in os.getenv("SCALP_PRECISION_GUARD_BYPASS_MODES", "").split(",")
    if s.strip()
}

# Performance / cooldown sync
PERF_REFRESH_SEC = float(os.getenv("SCALP_PRECISION_PERF_REFRESH_SEC", "30"))
STAGE_REFRESH_SEC = float(os.getenv("SCALP_PRECISION_STAGE_REFRESH_SEC", "15"))

# Entry guard based on account health (fail-open on missing values)
ENTRY_GUARD_ENABLED = _env_bool("SCALP_PRECISION_ENTRY_GUARD_ENABLED", True)
ENTRY_GUARD_MIN_FREE_MARGIN_RATIO = float(
    os.getenv(
        "SCALP_PRECISION_ENTRY_MIN_FREE_MARGIN_RATIO",
        os.getenv("EXIT_EMERGENCY_FREE_MARGIN_RATIO", "0.12"),
    )
)
ENTRY_GUARD_MAX_MARGIN_USAGE = float(
    os.getenv(
        "SCALP_PRECISION_ENTRY_MAX_MARGIN_USAGE",
        os.getenv("EXIT_EMERGENCY_MARGIN_USAGE_RATIO", "0.92"),
    )
)

# Entry drought guard (global entry gap filler)
DROUGHT_ENABLED = _env_bool("SCALP_PRECISION_DROUGHT_ENABLED", True)
DROUGHT_MINUTES = float(os.getenv("SCALP_PRECISION_DROUGHT_MINUTES", "15"))
DROUGHT_REFRESH_SEC = float(os.getenv("SCALP_PRECISION_DROUGHT_REFRESH_SEC", "15"))
DROUGHT_SOURCE = os.getenv("SCALP_PRECISION_DROUGHT_SOURCE", "trades").strip().lower()
DROUGHT_FAIL_OPEN = _env_bool("SCALP_PRECISION_DROUGHT_FAIL_OPEN", False)

# Drought mode tuning (looser range/revert gates)
DROUGHT_RANGE_SCORE = float(os.getenv("SCALP_PRECISION_DROUGHT_RANGE_SCORE", "0.38"))
DROUGHT_ADX_MAX = float(os.getenv("SCALP_PRECISION_DROUGHT_ADX_MAX", "26.0"))
# BBW is (upper-lower)/mid ratio (typical USD/JPY M1 ~= 0.0002..0.0020).
DROUGHT_BBW_MAX = float(os.getenv("SCALP_PRECISION_DROUGHT_BBW_MAX", "0.0016"))
DROUGHT_ATR_MIN = float(os.getenv("SCALP_PRECISION_DROUGHT_ATR_MIN", "0.5"))
DROUGHT_ATR_MAX = float(os.getenv("SCALP_PRECISION_DROUGHT_ATR_MAX", "4.0"))
DROUGHT_BB_TOUCH_PIPS = float(os.getenv("SCALP_PRECISION_DROUGHT_BB_TOUCH_PIPS", "1.0"))
DROUGHT_RSI_LONG_MAX = float(os.getenv("SCALP_PRECISION_DROUGHT_RSI_LONG_MAX", "49.0"))
DROUGHT_RSI_SHORT_MIN = float(os.getenv("SCALP_PRECISION_DROUGHT_RSI_SHORT_MIN", "51.0"))
DROUGHT_SPREAD_P25 = float(os.getenv("SCALP_PRECISION_DROUGHT_SPREAD_P25", "1.0"))

# Precision low-volatility mode tuning (strict entry filters)
PREC_LOWVOL_RANGE_SCORE = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_RANGE_SCORE", "0.25"))
PREC_LOWVOL_ADX_MAX = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_ADX_MAX", "30.0"))
PREC_LOWVOL_BBW_MAX = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_BBW_MAX", "0.0010"))
PREC_LOWVOL_ATR_MIN = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_ATR_MIN", "0.3"))
PREC_LOWVOL_ATR_MAX = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_ATR_MAX", "6.0"))
PREC_LOWVOL_BB_TOUCH_PIPS = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_BB_TOUCH_PIPS", "1.6"))
PREC_LOWVOL_RSI_LONG_MAX = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_RSI_LONG_MAX", "51.0"))
PREC_LOWVOL_RSI_SHORT_MIN = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_RSI_SHORT_MIN", "49.0"))
PREC_LOWVOL_STOCH_LONG_MAX = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_STOCH_LONG_MAX", "0.65"))
PREC_LOWVOL_STOCH_SHORT_MIN = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_STOCH_SHORT_MIN", "0.35"))
PREC_LOWVOL_VWAP_GAP_MIN = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_VWAP_GAP_MIN", "0.15"))
PREC_LOWVOL_VWAP_GAP_BLOCK = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_VWAP_GAP_BLOCK", "1.8"))
PREC_LOWVOL_SPREAD_P25 = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_SPREAD_P25", "1.8"))
PREC_LOWVOL_REV_MIN_STRENGTH = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_REV_MIN_STRENGTH", "0.28"))

# macd_rsi_div specific knobs
STRATEGY_TAG = os.getenv("SCALP_PRECISION_STRATEGY_TAG", "scalp_macd_rsi_div_live")
PIP_VALUE = float(os.getenv("SCALP_PRECISION_PIP_VALUE", "0.01"))
REQUIRE_RANGE_ACTIVE = _env_bool("SCALP_PRECISION_REQUIRE_RANGE_ACTIVE", False)
RANGE_MIN_SCORE = float(os.getenv("SCALP_PRECISION_RANGE_MIN_SCORE", "0.10"))
MAX_ADX = float(os.getenv("SCALP_PRECISION_MAX_ADX", "40"))
RSI_ARM_TTL_SEC = float(os.getenv("SCALP_PRECISION_RSI_ARM_TTL_SEC", "75"))
RSI_LONG_ARM = float(os.getenv("SCALP_PRECISION_RSI_LONG_ARM", "32"))
RSI_SHORT_ARM = float(os.getenv("SCALP_PRECISION_RSI_SHORT_ARM", "68"))
RSI_LONG_ENTRY = float(os.getenv("SCALP_PRECISION_RSI_LONG_ENTRY", "36"))
RSI_SHORT_ENTRY = float(os.getenv("SCALP_PRECISION_RSI_SHORT_ENTRY", "64"))
MIN_DIV_SCORE = float(os.getenv("SCALP_PRECISION_MIN_DIV_SCORE", "0.08"))
MIN_DIV_STRENGTH = float(os.getenv("SCALP_PRECISION_MIN_DIV_STRENGTH", "0.12"))
MAX_DIV_AGE_BARS = float(os.getenv("SCALP_PRECISION_MAX_DIV_AGE_BARS", "14"))
ALLOW_HIDDEN_DIVERGENCE = _env_bool("SCALP_PRECISION_ALLOW_HIDDEN_DIVERGENCE", True)
MIN_FREE_MARGIN_RATIO_HARD = float(os.getenv("SCALP_PRECISION_MIN_FREE_MARGIN_RATIO_HARD", "0.06"))
MARGIN_USAGE_HARD = float(os.getenv("SCALP_PRECISION_MARGIN_USAGE_HARD", "0.93"))
SL_ATR_MULT = float(os.getenv("SCALP_PRECISION_SL_ATR_MULT", "0.85"))
TP_ATR_MULT = float(os.getenv("SCALP_PRECISION_TP_ATR_MULT", "1.10"))
MAX_SL_PIPS = float(os.getenv("SCALP_PRECISION_MAX_SL_PIPS", "40"))
MIN_SL_PIPS = float(os.getenv("SCALP_PRECISION_MIN_SL_PIPS", "1"))
MAX_TP_PIPS = float(os.getenv("SCALP_PRECISION_MAX_TP_PIPS", "80"))
MIN_TP_PIPS = float(os.getenv("SCALP_PRECISION_MIN_TP_PIPS", "2"))
MIN_TP_RR = float(os.getenv("SCALP_PRECISION_MIN_TP_RR", "1.15"))
PATTERN_GATE_OPT_IN = _env_bool("SCALP_PRECISION_PATTERN_GATE_OPT_IN", True)
DYN_ALLOC_ENABLED = _env_bool("SCALP_PRECISION_DYN_ALLOC_ENABLED", False)
DYN_ALLOC_PATH = os.getenv("SCALP_PRECISION_DYN_ALLOC_PATH", "config/dynamic_alloc.json")
DYN_ALLOC_TTL_SEC = float(os.getenv("SCALP_PRECISION_DYN_ALLOC_TTL_SEC", "20"))
DYN_ALLOC_MIN_TRADES = int(float(os.getenv("SCALP_PRECISION_DYN_ALLOC_MIN_TRADES", "8")))
DYN_ALLOC_LOSER_SCORE = float(os.getenv("SCALP_PRECISION_DYN_ALLOC_LOSER_SCORE", "0.30"))
DYN_ALLOC_LOSER_BLOCK = _env_bool("SCALP_PRECISION_DYN_ALLOC_LOSER_BLOCK", True)
DYN_ALLOC_MULT_MIN = float(os.getenv("SCALP_PRECISION_DYN_ALLOC_MULT_MIN", "0.70"))
DYN_ALLOC_MULT_MAX = float(os.getenv("SCALP_PRECISION_DYN_ALLOC_MULT_MAX", "1.60"))
TECH_FAILOPEN = _env_bool("SCALP_PRECISION_TECH_FAILOPEN", True)
TECH_CONF_BOOST = float(os.getenv("SCALP_PRECISION_TECH_CONF_BOOST", "16.0"))
TECH_CONF_PENALTY = float(os.getenv("SCALP_PRECISION_TECH_CONF_PENALTY", "10.0"))
