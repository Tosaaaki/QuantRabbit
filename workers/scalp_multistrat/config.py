from __future__ import annotations

import os

_ENV_CACHE: dict | None = None


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

POCKET = "scalp"
LOOP_INTERVAL_SEC = float(os.getenv("SCALP_MULTI_LOOP_INTERVAL_SEC", "6.0"))
ENABLED = _env_bool("SCALP_MULTI_ENABLED", True)
LOG_PREFIX = os.getenv("SCALP_MULTI_LOG_PREFIX", "[ScalpMulti]")

CONFIDENCE_FLOOR = 30
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("SCALP_MULTI_MIN_UNITS", "1000"))
BASE_ENTRY_UNITS = int(os.getenv("SCALP_MULTI_BASE_UNITS", "6000"))
MAX_MARGIN_USAGE = float(os.getenv("SCALP_MULTI_MAX_MARGIN_USAGE", "0.9"))
AUTOTUNE_ENABLED = _env_bool("SCALP_AUTOTUNE_ENABLED", False)

CAP_MIN = float(os.getenv("SCALP_MULTI_CAP_MIN", "0.1"))
CAP_MAX = float(os.getenv("SCALP_MULTI_CAP_MAX", "0.9"))

# 新規エントリーのクールダウン（秒）
COOLDOWN_SEC = float(os.getenv("SCALP_MULTI_COOLDOWN_SEC", "90"))
# 同ポケットの最大同時建玉数
MAX_OPEN_TRADES = int(os.getenv("SCALP_MULTI_MAX_OPEN_TRADES", "2"))
# 経済指標ブロック（分）。未設定で AttributeError とならないようデフォルトを明示。
NEWS_BLOCK_MINUTES = float(os.getenv("SCALP_MULTI_NEWS_BLOCK_MINUTES", "0"))

# Range-mode selection bias.
RANGE_ONLY_SCORE = float(os.getenv("SCALP_MULTI_RANGE_ONLY_SCORE", "0.45"))
RANGE_BIAS_SCORE = float(os.getenv("SCALP_MULTI_RANGE_BIAS_SCORE", "0.30"))
RANGE_STRATEGY_BONUS = float(os.getenv("SCALP_MULTI_RANGE_STRATEGY_BONUS", "12"))
RANGE_TREND_PENALTY = float(os.getenv("SCALP_MULTI_RANGE_TREND_PENALTY", "10"))

# Strategy diversity: promote idle strategies without inflating risk sizing.
DIVERSITY_ENABLED = _env_bool("SCALP_MULTI_DIVERSITY_ENABLED", True)
DIVERSITY_IDLE_SEC = float(os.getenv("SCALP_MULTI_DIVERSITY_IDLE_SEC", "180"))
DIVERSITY_SCALE_SEC = float(os.getenv("SCALP_MULTI_DIVERSITY_SCALE_SEC", "600"))
DIVERSITY_MAX_BONUS = float(os.getenv("SCALP_MULTI_DIVERSITY_MAX_BONUS", "10"))

# Multi-signal dispatch (variety): send top-N signals per cycle with smaller sizing.
MAX_SIGNALS_PER_CYCLE = int(os.getenv("SCALP_MULTI_MAX_SIGNALS_PER_CYCLE", "2"))
MULTI_SIGNAL_MIN_SCALE = float(os.getenv("SCALP_MULTI_MULTI_SIGNAL_MIN_SCALE", "0.6"))
