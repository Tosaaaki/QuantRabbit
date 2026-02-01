from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


POCKET = "scalp"
LOOP_INTERVAL_SEC = float(os.getenv("RANGEFADER_LOOP_INTERVAL_SEC", "3.0"))
ENABLED = _env_bool("RANGEFADER_ENABLED", True)
LOG_PREFIX = os.getenv("RANGEFADER_LOG_PREFIX", "[RangeFader]")

CONFIDENCE_FLOOR = 25
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("RANGEFADER_MIN_UNITS", "2000"))
BASE_ENTRY_UNITS = int(os.getenv("RANGEFADER_BASE_UNITS", "13000"))

CAP_MIN = float(os.getenv("RANGEFADER_CAP_MIN", "0.1"))
CAP_MAX = float(os.getenv("RANGEFADER_CAP_MAX", "1.0"))

COOLDOWN_SEC = float(os.getenv("RANGEFADER_COOLDOWN_SEC", "30.0"))
MAX_OPEN_TRADES = int(os.getenv("RANGEFADER_MAX_OPEN_TRADES", "4"))
MAX_OPEN_TRADES_GLOBAL = int(os.getenv("RANGEFADER_MAX_OPEN_TRADES_GLOBAL", "0"))
OPEN_TRADES_SCOPE = os.getenv("RANGEFADER_OPEN_TRADES_SCOPE", "tag").strip().lower()
RANGE_ONLY_SCORE = float(os.getenv("RANGEFADER_RANGE_ONLY_SCORE", "0.40"))

# Size boost for strong range setups
SIZE_MULT_BASE = float(os.getenv("RANGEFADER_SIZE_MULT_BASE", "1.25"))
SIZE_MULT_SCORE_START = float(os.getenv("RANGEFADER_SIZE_MULT_SCORE_START", "0.45"))
SIZE_MULT_SLOPE = float(os.getenv("RANGEFADER_SIZE_MULT_SLOPE", "0.7"))
SIZE_MULT_MIN = float(os.getenv("RANGEFADER_SIZE_MULT_MIN", "0.9"))
SIZE_MULT_MAX = float(os.getenv("RANGEFADER_SIZE_MULT_MAX", "1.5"))
SIZE_MULT_MIN_FMR = float(os.getenv("RANGEFADER_SIZE_MULT_MIN_FMR", "0.12"))
