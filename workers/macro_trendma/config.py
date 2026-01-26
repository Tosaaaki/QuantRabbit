from __future__ import annotations

import os

POCKET = "macro"
LOOP_INTERVAL_SEC = float(os.getenv("TRENDMA_LOOP_INTERVAL_SEC", "15.0"))
ENABLED = os.getenv("TRENDMA_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[TrendMA]"

CONFIDENCE_FLOOR = 40
CONFIDENCE_CEIL = 90
CONFIDENCE_MIN_BASE = int(os.getenv("TRENDMA_CONFIDENCE_MIN_BASE", "70"))
CONFIDENCE_MIN_HIGH_VOL = int(os.getenv("TRENDMA_CONFIDENCE_MIN_HIGH_VOL", "85"))
CONFIDENCE_MIN_ATR_PIPS = float(os.getenv("TRENDMA_CONFIDENCE_MIN_ATR_PIPS", "8.0"))
MIN_UNITS = int(os.getenv("TRENDMA_MIN_UNITS", "2000"))
BASE_ENTRY_UNITS = int(os.getenv("TRENDMA_BASE_UNITS", "35000"))
MAX_MARGIN_USAGE = float(os.getenv("TRENDMA_MAX_MARGIN_USAGE", "0.92"))

CAP_MIN = float(os.getenv("TRENDMA_CAP_MIN", "0.25"))
CAP_MAX = float(os.getenv("TRENDMA_CAP_MAX", "0.99"))
