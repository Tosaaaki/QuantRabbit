from __future__ import annotations

import os

POCKET = "macro"
LOOP_INTERVAL_SEC = float(os.getenv("DON55_LOOP_INTERVAL_SEC", "20.0"))
ENABLED = os.getenv("DON55_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[Donchian55]"

CONFIDENCE_FLOOR = 45
CONFIDENCE_CEIL = 95
MIN_UNITS = int(os.getenv("DON55_MIN_UNITS", "2000"))
BASE_ENTRY_UNITS = int(os.getenv("DON55_BASE_UNITS", "30000"))
MAX_MARGIN_USAGE = float(os.getenv("DON55_MAX_MARGIN_USAGE", "0.9"))

CAP_MIN = float(os.getenv("DON55_CAP_MIN", "0.2"))
CAP_MAX = float(os.getenv("DON55_CAP_MAX", "0.8"))

