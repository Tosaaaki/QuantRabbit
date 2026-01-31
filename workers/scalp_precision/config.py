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

MIN_UNITS = int(float(os.getenv("SCALP_PRECISION_MIN_UNITS", "1000")))
BASE_ENTRY_UNITS = int(float(os.getenv("SCALP_PRECISION_BASE_UNITS", "9000")))
MAX_MARGIN_USAGE = float(os.getenv("SCALP_PRECISION_MAX_MARGIN_USAGE", "0.92"))

CAP_MIN = float(os.getenv("SCALP_PRECISION_CAP_MIN", "0.12"))
CAP_MAX = float(os.getenv("SCALP_PRECISION_CAP_MAX", "0.95"))

COOLDOWN_SEC = float(os.getenv("SCALP_PRECISION_COOLDOWN_SEC", "45"))
MAX_OPEN_TRADES = int(float(os.getenv("SCALP_PRECISION_MAX_OPEN_TRADES", "2")))

MAX_SPREAD_PIPS = float(os.getenv("SCALP_PRECISION_MAX_SPREAD_PIPS", "1.2"))

MAX_SIGNALS_PER_CYCLE = int(float(os.getenv("SCALP_PRECISION_MAX_SIGNALS_PER_CYCLE", "1")))

MODE = os.getenv("SCALP_PRECISION_MODE", "spread_revert").strip().lower()
ALLOWLIST_RAW = os.getenv("SCALP_PRECISION_ALLOWLIST", "").strip()
