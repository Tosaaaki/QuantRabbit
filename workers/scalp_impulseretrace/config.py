from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


POCKET = "scalp"
LOOP_INTERVAL_SEC = float(os.getenv("IMPULSERETRACE_LOOP_INTERVAL_SEC", "6.0"))
ENABLED = _env_bool("IMPULSERETRACE_ENABLED", True)
LOG_PREFIX = os.getenv("IMPULSERETRACE_LOG_PREFIX", "[ImpulseRetrace]")

CONFIDENCE_FLOOR = 30
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("IMPULSERETRACE_MIN_UNITS", "1000"))
BASE_ENTRY_UNITS = int(os.getenv("IMPULSERETRACE_BASE_UNITS", "6000"))

CAP_MIN = float(os.getenv("IMPULSERETRACE_CAP_MIN", "0.1"))
CAP_MAX = float(os.getenv("IMPULSERETRACE_CAP_MAX", "0.9"))

COOLDOWN_SEC = float(os.getenv("IMPULSERETRACE_COOLDOWN_SEC", "90.0"))
MAX_OPEN_TRADES = int(os.getenv("IMPULSERETRACE_MAX_OPEN_TRADES", "2"))
RANGE_ONLY_SCORE = float(os.getenv("IMPULSERETRACE_RANGE_ONLY_SCORE", "0.45"))
