from __future__ import annotations

import os

POCKET = "micro"
LOOP_INTERVAL_SEC = float(os.getenv("MICRO_MULTI_LOOP_INTERVAL_SEC", "8.0"))
ENABLED = os.getenv("MICRO_MULTI_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[MicroMulti]"

CONFIDENCE_FLOOR = 35
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("MICRO_MULTI_MIN_UNITS", "1500"))
BASE_ENTRY_UNITS = int(os.getenv("MICRO_MULTI_BASE_UNITS", "12000"))
MAX_MARGIN_USAGE = float(os.getenv("MICRO_MULTI_MAX_MARGIN_USAGE", "0.9"))

CAP_MIN = float(os.getenv("MICRO_MULTI_CAP_MIN", "0.15"))
CAP_MAX = float(os.getenv("MICRO_MULTI_CAP_MAX", "0.8"))

