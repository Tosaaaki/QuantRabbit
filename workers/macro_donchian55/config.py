from __future__ import annotations

import os

POCKET = "macro"
LOOP_INTERVAL_SEC = float(os.getenv("DON55_LOOP_INTERVAL_SEC", "20.0"))
ENABLED = os.getenv("DON55_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[Donchian55]"

CONFIDENCE_FLOOR = 45
CONFIDENCE_CEIL = 95
CONFIDENCE_MIN_BASE = int(os.getenv("DON55_CONFIDENCE_MIN_BASE", "70"))
CONFIDENCE_MIN_HIGH_VOL = int(os.getenv("DON55_CONFIDENCE_MIN_HIGH_VOL", "85"))
CONFIDENCE_MIN_ATR_PIPS = float(os.getenv("DON55_CONFIDENCE_MIN_ATR_PIPS", "8.0"))
MIN_FREE_MARGIN_RATIO = float(os.getenv("DON55_MIN_FREE_MARGIN_RATIO", "0.10"))
MIN_UNITS = int(os.getenv("DON55_MIN_UNITS", "2000"))
BASE_ENTRY_UNITS = int(os.getenv("DON55_BASE_UNITS", "30000"))
MAX_MARGIN_USAGE = float(os.getenv("DON55_MAX_MARGIN_USAGE", "0.9"))

CAP_MIN = float(os.getenv("DON55_CAP_MIN", "0.2"))
CAP_MAX = float(os.getenv("DON55_CAP_MAX", "0.95"))
