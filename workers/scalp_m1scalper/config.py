from __future__ import annotations

import os

_ENV_CACHE: dict | None = None
ENV_PREFIX = "M1SCALP"


def _load_env_file() -> dict:
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    data: dict = {}
    path = os.getenv("QUANTRABBIT_ENV_FILE", "/etc/quantrabbit.env")
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

POCKET = "scalp"
LOOP_INTERVAL_SEC = float(os.getenv("M1SCALP_LOOP_INTERVAL_SEC", "6.0"))
ENABLED = _env_bool("M1SCALP_ENABLED", True)
LOG_PREFIX = "[M1Scalper]"

CONFIDENCE_FLOOR = 50
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("M1SCALP_MIN_UNITS", "1000"))
BASE_ENTRY_UNITS = int(os.getenv("M1SCALP_BASE_UNITS", "6000"))
MAX_MARGIN_USAGE = float(os.getenv("M1SCALP_MAX_MARGIN_USAGE", "0.9"))
MAX_SPREAD_PIPS = float(os.getenv("M1SCALP_MAX_SPREAD_PIPS", "1.4"))
AUTOTUNE_ENABLED = _env_bool("SCALP_AUTOTUNE_ENABLED", False)

CAP_MIN = float(os.getenv("M1SCALP_CAP_MIN", "0.1"))
CAP_MAX = float(os.getenv("M1SCALP_CAP_MAX", "0.9"))
# Higher timeframe trend guard (H1)
HTF_ADX_MIN = float(os.getenv("M1SCALP_HTF_ADX_MIN", "23"))
HTF_GAP_PIPS = float(os.getenv("M1SCALP_HTF_GAP_PIPS", "6.0"))
HTF_BLOCK_COUNTER = _env_bool("M1SCALP_HTF_BLOCK_COUNTER", True)
# Projection flip (direction correction without reducing entry count)
PROJ_FLIP_ENABLED = _env_bool("M1SCALP_PROJ_FLIP_ENABLED", True)
PROJ_FLIP_MARGIN = float(os.getenv("M1SCALP_PROJ_FLIP_MARGIN", "0.10"))
PROJ_FLIP_MIN_SCORE = float(os.getenv("M1SCALP_PROJ_FLIP_MIN_SCORE", "0.10"))
PROJ_FLIP_MAX_SCORE = float(os.getenv("M1SCALP_PROJ_FLIP_MAX_SCORE", "-0.05"))
# Mean-reversion environment guard (tick gaps / return spikes)
ENV_GUARD_ENABLED = _env_bool("M1SCALP_ENV_GUARD_ENABLED", True)
ENV_SPREAD_P50_LIMIT = float(os.getenv("M1SCALP_ENV_SPREAD_P50_LIMIT", "0.35"))
ENV_RETURN_PIPS_LIMIT = float(os.getenv("M1SCALP_ENV_RETURN_PIPS_LIMIT", "4.5"))
ENV_RETURN_WINDOW_SEC = float(os.getenv("M1SCALP_ENV_RETURN_WINDOW_SEC", "12.0"))
ENV_INSTANT_MOVE_LIMIT = float(os.getenv("M1SCALP_ENV_INSTANT_MOVE_LIMIT", "1.8"))
ENV_TICK_GAP_MS_LIMIT = float(os.getenv("M1SCALP_ENV_TICK_GAP_MS_LIMIT", "900"))
ENV_TICK_GAP_MOVE_PIPS = float(os.getenv("M1SCALP_ENV_TICK_GAP_MOVE_PIPS", "1.2"))
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
