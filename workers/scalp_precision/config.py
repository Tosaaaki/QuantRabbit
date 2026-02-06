from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


POCKET = "scalp"
ENABLED = _env_bool("SCALP_PRECISION_ENABLED", False)
LOOP_INTERVAL_SEC = float(os.getenv("SCALP_PRECISION_LOOP_INTERVAL_SEC", "4.0"))
LOG_PREFIX = os.getenv("SCALP_PRECISION_LOG_PREFIX", "[ScalpPrecision]")

CONFIDENCE_FLOOR = int(float(os.getenv("SCALP_PRECISION_CONF_FLOOR", "32")))
CONFIDENCE_CEIL = int(float(os.getenv("SCALP_PRECISION_CONF_CEIL", "92")))
MIN_ENTRY_CONF = int(float(os.getenv("SCALP_PRECISION_MIN_ENTRY_CONF", str(CONFIDENCE_FLOOR))))

MIN_UNITS = int(float(os.getenv("SCALP_PRECISION_MIN_UNITS", "1000")))
# /etc/quantrabbit.env may define global sizing for the whole pocket. For per-unit tuning,
# prefer an explicit unit override so a single strategy can be resized without editing the
# global env file.
BASE_ENTRY_UNITS = int(
    float(
        os.getenv(
            "SCALP_PRECISION_UNIT_BASE_UNITS",
            os.getenv("SCALP_PRECISION_BASE_UNITS", "9000"),
        )
    )
)
MAX_MARGIN_USAGE = float(os.getenv("SCALP_PRECISION_MAX_MARGIN_USAGE", "0.92"))

CAP_MIN = float(os.getenv("SCALP_PRECISION_CAP_MIN", "0.12"))
CAP_MAX = float(
    os.getenv(
        "SCALP_PRECISION_UNIT_CAP_MAX",
        os.getenv("SCALP_PRECISION_CAP_MAX", "0.95"),
    )
)

COOLDOWN_SEC = float(os.getenv("SCALP_PRECISION_COOLDOWN_SEC", "45"))
MAX_OPEN_TRADES = int(float(os.getenv("SCALP_PRECISION_MAX_OPEN_TRADES", "2")))
MAX_OPEN_TRADES_GLOBAL = int(float(os.getenv("SCALP_PRECISION_MAX_OPEN_TRADES_GLOBAL", "0")))
OPEN_TRADES_SCOPE = os.getenv("SCALP_PRECISION_OPEN_TRADES_SCOPE", "tag").strip().lower()

MAX_SPREAD_PIPS = float(os.getenv("SCALP_PRECISION_MAX_SPREAD_PIPS", "1.2"))

MAX_SIGNALS_PER_CYCLE = int(float(os.getenv("SCALP_PRECISION_MAX_SIGNALS_PER_CYCLE", "1")))

MODE = os.getenv("SCALP_PRECISION_MODE", "spread_revert").strip().lower()
# /etc/quantrabbit.env may define a global allowlist shared by multiple units. Prefer an explicit
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
DROUGHT_BBW_MAX = float(os.getenv("SCALP_PRECISION_DROUGHT_BBW_MAX", "0.28"))
DROUGHT_ATR_MIN = float(os.getenv("SCALP_PRECISION_DROUGHT_ATR_MIN", "0.5"))
DROUGHT_ATR_MAX = float(os.getenv("SCALP_PRECISION_DROUGHT_ATR_MAX", "4.0"))
DROUGHT_BB_TOUCH_PIPS = float(os.getenv("SCALP_PRECISION_DROUGHT_BB_TOUCH_PIPS", "1.0"))
DROUGHT_RSI_LONG_MAX = float(os.getenv("SCALP_PRECISION_DROUGHT_RSI_LONG_MAX", "49.0"))
DROUGHT_RSI_SHORT_MIN = float(os.getenv("SCALP_PRECISION_DROUGHT_RSI_SHORT_MIN", "51.0"))
DROUGHT_SPREAD_P25 = float(os.getenv("SCALP_PRECISION_DROUGHT_SPREAD_P25", "1.0"))

# Precision low-volatility mode tuning (strict entry filters)
PREC_LOWVOL_RANGE_SCORE = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_RANGE_SCORE", "0.25"))
PREC_LOWVOL_ADX_MAX = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_ADX_MAX", "30.0"))
PREC_LOWVOL_BBW_MAX = float(os.getenv("SCALP_PRECISION_PREC_LOWVOL_BBW_MAX", "0.38"))
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
