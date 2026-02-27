from __future__ import annotations

import os

_ENV_CACHE: dict | None = None
ENV_PREFIX = "M1SCALP"


def _load_env_file() -> dict:
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    data: dict = {}
    path = os.getenv(
        "QUANTRABBIT_ENV_FILE",
        "/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env",
    )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                data[key.strip()] = val.strip().strip('"').strip("'")
    except OSError:
        pass
    _ENV_CACHE = data
    return data


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        raw = _load_env_file().get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        raw = _load_env_file().get(name)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)

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

def _parse_csv_lower(raw: str) -> set[str]:
    items: set[str] = set()
    for token in str(raw or "").split(","):
        t = token.strip().lower()
        if t:
            items.add(t)
    return items


def _normalize_side(raw: str) -> str:
    side = str(raw or "").strip().lower()
    if side in {"buy", "long", "open_long"}:
        return "long"
    if side in {"sell", "short", "open_short"}:
        return "short"
    return ""


POCKET = "scalp"
LOOP_INTERVAL_SEC = float(os.getenv("M1SCALP_LOOP_INTERVAL_SEC", "6.0"))
ENABLED = _env_bool("M1SCALP_ENABLED", True)
LOG_PREFIX = "[TrendBreakout]"

CONFIDENCE_FLOOR = int(os.getenv("M1SCALP_CONFIDENCE_FLOOR", "50"))
CONFIDENCE_CEIL = int(os.getenv("M1SCALP_CONFIDENCE_CEIL", "90"))
MIN_UNITS = int(os.getenv("M1SCALP_MIN_UNITS", "1000"))
BASE_ENTRY_UNITS = int(os.getenv("M1SCALP_BASE_UNITS", "6000"))
MAX_MARGIN_USAGE = float(os.getenv("M1SCALP_MAX_MARGIN_USAGE", "0.9"))
# Emergency brake: hard-stop new M1Scalper entries when margin health is poor.
MIN_FREE_MARGIN_RATIO_HARD = _env_float("M1SCALP_MIN_FREE_MARGIN_RATIO_HARD", 0.08)
MARGIN_USAGE_HARD = _env_float("M1SCALP_MARGIN_USAGE_HARD", 0.93)
MAX_SPREAD_PIPS = float(os.getenv("M1SCALP_MAX_SPREAD_PIPS", "1.4"))
# Default: rely on spread_monitor as the single spread guard source.
# Keep local cap available as an explicit fallback (or when guard is disabled).
LOCAL_SPREAD_CAP_ENABLED = _env_bool("M1SCALP_LOCAL_SPREAD_CAP_ENABLED", False)
AUTOTUNE_ENABLED = _env_bool("SCALP_AUTOTUNE_ENABLED", False)
ENTRY_GUARD_BYPASS = _env_bool("M1SCALP_ENTRY_GUARD_BYPASS", False)

CAP_MIN = float(os.getenv("M1SCALP_CAP_MIN", "0.1"))
CAP_MAX = float(os.getenv("M1SCALP_CAP_MAX", "0.9"))
# Higher timeframe trend guard (H1)
HTF_ADX_MIN = float(os.getenv("M1SCALP_HTF_ADX_MIN", "23"))
HTF_GAP_PIPS = float(os.getenv("M1SCALP_HTF_GAP_PIPS", "6.0"))
HTF_BLOCK_COUNTER = _env_bool("M1SCALP_HTF_BLOCK_COUNTER", True)
BB_ENTRY_SOFT_FAIL_ENABLED = _env_bool("M1SCALP_BB_ENTRY_SOFT_FAIL_ENABLED", True)
BB_ENTRY_SOFT_FAIL_RATIO = _env_float("M1SCALP_BB_ENTRY_SOFT_FAIL_RATIO", 0.25)
BB_ENTRY_SOFT_FAIL_MIN_SCALE = _env_float("M1SCALP_BB_ENTRY_SOFT_FAIL_MIN_SCALE", 0.56)
BB_ENTRY_SOFT_FAIL_MAX_SCALE = _env_float("M1SCALP_BB_ENTRY_SOFT_FAIL_MAX_SCALE", 0.86)
BB_ENTRY_MIN_CONF_FOR_SOFT_FAIL = int(os.getenv("M1SCALP_BB_ENTRY_MIN_CONF_FOR_SOFT_FAIL", "66"))
BB_MID_TOLERANCE_PIPS = _env_float("M1SCALP_BB_MID_TOLERANCE_PIPS", 0.55)
# Projection flip (direction correction without reducing entry count)
PROJ_FLIP_ENABLED = _env_bool("M1SCALP_PROJ_FLIP_ENABLED", True)
PROJ_FLIP_MARGIN = float(os.getenv("M1SCALP_PROJ_FLIP_MARGIN", "0.10"))
PROJ_FLIP_MIN_SCORE = float(os.getenv("M1SCALP_PROJ_FLIP_MIN_SCORE", "0.10"))
PROJ_FLIP_MAX_SCORE = float(os.getenv("M1SCALP_PROJ_FLIP_MAX_SCORE", "-0.05"))
PROJ_SOFT_FAIL_ENABLED = _env_bool("M1SCALP_PROJ_SOFT_FAIL_ENABLED", True)
PROJ_SOFT_FAIL_SCORE = float(os.getenv("M1SCALP_PROJ_SOFT_FAIL_SCORE", "-0.60"))
PROJ_SOFT_FAIL_MIN_SCALE = _env_float("M1SCALP_PROJ_SOFT_FAIL_MIN_SCALE", 0.65)
PROJ_SOFT_FAIL_MAX_SCALE = _env_float("M1SCALP_PROJ_SOFT_FAIL_MAX_SCALE", 0.90)
# Mean-reversion environment guard (tick gaps / return spikes)
ENV_GUARD_ENABLED = _env_bool("M1SCALP_ENV_GUARD_ENABLED", True)
ENV_SPREAD_P50_LIMIT = float(os.getenv("M1SCALP_ENV_SPREAD_P50_LIMIT", "0.35"))
ENV_RETURN_PIPS_LIMIT = float(os.getenv("M1SCALP_ENV_RETURN_PIPS_LIMIT", "4.5"))
ENV_RETURN_WINDOW_SEC = float(os.getenv("M1SCALP_ENV_RETURN_WINDOW_SEC", "12.0"))
ENV_INSTANT_MOVE_LIMIT = float(os.getenv("M1SCALP_ENV_INSTANT_MOVE_LIMIT", "1.8"))
ENV_TICK_GAP_MS_LIMIT = float(os.getenv("M1SCALP_ENV_TICK_GAP_MS_LIMIT", "900"))
ENV_TICK_GAP_MOVE_PIPS = float(os.getenv("M1SCALP_ENV_TICK_GAP_MOVE_PIPS", "1.2"))
MIN_UNITS_SOFT_FAIL_ENABLED = _env_bool("M1SCALP_MIN_UNITS_SOFT_FAIL_ENABLED", True)
MIN_UNITS_SOFT_FAIL_RATIO = _env_float("M1SCALP_MIN_UNITS_SOFT_FAIL_RATIO", 0.72)
CANDLE_SOFT_FAIL_ENABLED = _env_bool("M1SCALP_CANDLE_SOFT_FAIL_ENABLED", True)
CANDLE_SOFT_FAIL_SCORE = float(os.getenv("M1SCALP_CANDLE_SOFT_FAIL_SCORE", "-0.70"))
CANDLE_SOFT_FAIL_MIN_SCALE = _env_float("M1SCALP_CANDLE_SOFT_FAIL_MIN_SCALE", 0.45)
CANDLE_SOFT_FAIL_MAX_SCALE = _env_float("M1SCALP_CANDLE_SOFT_FAIL_MAX_SCALE", 0.82)
CANDLE_MIN_CONF = _env_float("M1SCALP_CANDLE_MIN_CONF", 0.42)
CANDLE_ENTRY_BLOCK = _env_float("M1SCALP_CANDLE_ENTRY_BLOCK", -0.6)
CANDLE_ENTRY_SCALE = _env_float("M1SCALP_CANDLE_ENTRY_SCALE", 0.2)
CANDLE_ENTRY_MIN = _env_float("M1SCALP_CANDLE_ENTRY_MIN", 0.8)
CANDLE_ENTRY_MAX = _env_float("M1SCALP_CANDLE_ENTRY_MAX", 1.2)
TECH_SOFT_FAIL_ENABLED = _env_bool("M1SCALP_TECH_SOFT_FAIL_ENABLED", True)
TECH_SOFT_FAIL_SCORE = float(os.getenv("M1SCALP_TECH_SOFT_FAIL_SCORE", "-0.05"))
TECH_SOFT_FAIL_MIN_SCALE = _env_float("M1SCALP_TECH_SOFT_FAIL_MIN_SCALE", 0.45)
TECH_SOFT_FAIL_MAX_SCALE = _env_float("M1SCALP_TECH_SOFT_FAIL_MAX_SCALE", 0.88)
# クールダウン/同時建玉制御（デフォルトを明示）
COOLDOWN_SEC = float(os.getenv("M1SCALP_COOLDOWN_SEC", "120"))
MAX_OPEN_TRADES = int(os.getenv("M1SCALP_MAX_OPEN_TRADES", "1"))
# 直近の大敗時間帯を任意でブロック（デフォルト無効）
BLOCK_HOURS_UTC = frozenset(
    _parse_hours(os.getenv("M1SCALP_BLOCK_HOURS_UTC", ""))
)
BLOCK_HOURS_ENABLED = os.getenv("M1SCALP_BLOCK_HOURS_ENABLED", "0").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}

# Optional regime filter: comma-separated {"range","trend","breakout","mixed"} etc.
# Empty = allow all (preserves current behavior).
ALLOWED_REGIMES = frozenset(
    _parse_csv_lower(
        os.getenv("M1SCALP_ALLOWED_REGIMES")
        or _load_env_file().get("M1SCALP_ALLOWED_REGIMES", "")
    )
)

# Dynamic winner routing from config/dynamic_alloc.json
DYN_ALLOC_ENABLED = _env_bool("M1SCALP_DYN_ALLOC_ENABLED", True)
DYN_ALLOC_PATH = os.getenv("M1SCALP_DYN_ALLOC_PATH", "config/dynamic_alloc.json")
DYN_ALLOC_TTL_SEC = float(os.getenv("M1SCALP_DYN_ALLOC_TTL_SEC", "20"))
DYN_ALLOC_MIN_TRADES = int(os.getenv("M1SCALP_DYN_ALLOC_MIN_TRADES", "8"))
DYN_ALLOC_LOSER_SCORE = float(os.getenv("M1SCALP_DYN_ALLOC_LOSER_SCORE", "0.30"))
DYN_ALLOC_LOSER_BLOCK = _env_bool("M1SCALP_DYN_ALLOC_LOSER_BLOCK", True)
DYN_ALLOC_MULT_MIN = float(os.getenv("M1SCALP_DYN_ALLOC_MULT_MIN", "0.7"))
DYN_ALLOC_MULT_MAX = float(os.getenv("M1SCALP_DYN_ALLOC_MULT_MAX", "1.8"))

# Entry style gating
# Reversion leg (buy-dip / sell-rally) was a recurring loss source in production.
# Keep it disabled by default and allow explicit opt-in via env.
ALLOW_REVERSION = _env_bool("M1SCALP_ALLOW_REVERSION", False)
ALLOW_TREND = _env_bool("M1SCALP_ALLOW_TREND", True)
# Optional signal filter for derivative strategy workers.
# e.g. M1SCALP_SIGNAL_TAG_CONTAINS=trend-long
SIGNAL_TAG_CONTAINS = frozenset(
    _parse_csv_lower(
        os.getenv("M1SCALP_SIGNAL_TAG_CONTAINS")
        or _load_env_file().get("M1SCALP_SIGNAL_TAG_CONTAINS", "breakout-retest")
    )
)
# USDJPY directional setup gating / sizing (strategy level, optional).
USDJPY_SETUP_GATING = _env_bool("M1SCALP_USDJPY_SETUP_GATING", False)
USDJPY_PULLBACK_BAND_PIPS = _env_float("M1SCALP_USDJPY_PULLBACK_BAND_PIPS", 2.6)
USDJPY_BREAK_LOOKBACK_M1 = int(_env_float("M1SCALP_USDJPY_BREAK_LOOKBACK_M1", 18))
USDJPY_BREAK_LOOKBACK_M5 = int(_env_float("M1SCALP_USDJPY_BREAK_LOOKBACK_M5", 12))
USDJPY_BREAK_MARGIN_PIPS = _env_float("M1SCALP_USDJPY_BREAK_MARGIN_PIPS", 1.4)
USDJPY_PULLBACK_SIZE_MULT = _env_float("M1SCALP_USDJPY_PULLBACK_SIZE_MULT", 0.98)
USDJPY_BREAK_SIZE_MULT = _env_float("M1SCALP_USDJPY_BREAK_SIZE_MULT", 1.05)
# Optional "100 JPY quickshot" profile:
# - align with breakout_retest setup
# - require M5 breakout + M1 pullback
# - derive TP/SL from ATR and reverse-size units toward a JPY target
USDJPY_QUICKSHOT_ENABLED = _env_bool("M1SCALP_USDJPY_QUICKSHOT_ENABLED", False)
USDJPY_QUICKSHOT_REQUIRE_BREAKOUT_RETEST = _env_bool(
    "M1SCALP_USDJPY_QUICKSHOT_REQUIRE_BREAKOUT_RETEST",
    True,
)
USDJPY_QUICKSHOT_TARGET_JPY = _env_float("M1SCALP_USDJPY_QUICKSHOT_TARGET_JPY", 100.0)
USDJPY_QUICKSHOT_EST_COST_JPY = _env_float("M1SCALP_USDJPY_QUICKSHOT_EST_COST_JPY", 12.0)
USDJPY_QUICKSHOT_MAX_SPREAD_PIPS = _env_float("M1SCALP_USDJPY_QUICKSHOT_MAX_SPREAD_PIPS", 0.30)
USDJPY_QUICKSHOT_BREAK_LOOKBACK_M5 = int(_env_float("M1SCALP_USDJPY_QUICKSHOT_BREAK_LOOKBACK_M5", 12))
USDJPY_QUICKSHOT_BREAK_MARGIN_PIPS = _env_float("M1SCALP_USDJPY_QUICKSHOT_BREAK_MARGIN_PIPS", 0.25)
USDJPY_QUICKSHOT_PULLBACK_BAND_PIPS = _env_float("M1SCALP_USDJPY_QUICKSHOT_PULLBACK_BAND_PIPS", 2.2)
USDJPY_QUICKSHOT_RETRACE_MIN_PIPS = _env_float("M1SCALP_USDJPY_QUICKSHOT_RETRACE_MIN_PIPS", 0.35)
USDJPY_QUICKSHOT_TP_ATR_MULT = _env_float("M1SCALP_USDJPY_QUICKSHOT_TP_ATR_MULT", 2.1)
USDJPY_QUICKSHOT_TP_PIPS_MIN = _env_float("M1SCALP_USDJPY_QUICKSHOT_TP_PIPS_MIN", 6.0)
USDJPY_QUICKSHOT_TP_PIPS_MAX = _env_float("M1SCALP_USDJPY_QUICKSHOT_TP_PIPS_MAX", 12.0)
USDJPY_QUICKSHOT_SL_TP_RATIO = _env_float("M1SCALP_USDJPY_QUICKSHOT_SL_TP_RATIO", 0.60)
USDJPY_QUICKSHOT_SL_PIPS_MIN = _env_float("M1SCALP_USDJPY_QUICKSHOT_SL_PIPS_MIN", 3.5)
USDJPY_QUICKSHOT_SL_PIPS_MAX = _env_float("M1SCALP_USDJPY_QUICKSHOT_SL_PIPS_MAX", 9.0)
USDJPY_QUICKSHOT_MIN_ENTRY_PROBABILITY = _env_float("M1SCALP_USDJPY_QUICKSHOT_MIN_ENTRY_PROBABILITY", 0.55)
USDJPY_QUICKSHOT_BLOCK_JST_HOURS = frozenset(
    _parse_hours(
        os.getenv("M1SCALP_USDJPY_QUICKSHOT_BLOCK_JST_HOURS")
        or _load_env_file().get("M1SCALP_USDJPY_QUICKSHOT_BLOCK_JST_HOURS", "7")
    )
)
# Optional side filter: long / short (empty = both).
SIDE_FILTER = _normalize_side(
    os.getenv("M1SCALP_SIDE_FILTER")
    or _load_env_file().get("M1SCALP_SIDE_FILTER", "")
)
# Optional strategy tag override for downstream tracking.
STRATEGY_TAG_OVERRIDE = (
    os.getenv("M1SCALP_STRATEGY_TAG_OVERRIDE")
    or _load_env_file().get("M1SCALP_STRATEGY_TAG_OVERRIDE", "TrendBreakout")
).strip()

# Reversion safety gate: even if ALLOW_REVERSION=1, require robust range context by default.
REVERSION_REQUIRE_STRONG_RANGE = _env_bool("M1SCALP_REVERSION_REQUIRE_STRONG_RANGE", True)
REVERSION_MIN_RANGE_SCORE = float(os.getenv("M1SCALP_REVERSION_MIN_RANGE_SCORE", "0.68"))
REVERSION_MAX_ADX = float(os.getenv("M1SCALP_REVERSION_MAX_ADX", "20.0"))
REVERSION_ALLOWED_RANGE_MODES = frozenset(
    _parse_csv_lower(
        os.getenv("M1SCALP_REVERSION_ALLOWED_RANGE_MODES")
        or _load_env_file().get("M1SCALP_REVERSION_ALLOWED_RANGE_MODES", "range")
    )
)
