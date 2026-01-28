from __future__ import annotations

import os

POCKET = "micro"
LOOP_INTERVAL_SEC = float(os.getenv("MICRO_RB_LOOP_INTERVAL_SEC", "8.0"))
ENABLED = os.getenv("MICRO_RB_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[MicroRangeBreak]"

CONFIDENCE_FLOOR = 35
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("MICRO_RB_MIN_UNITS", "1500"))
BASE_ENTRY_UNITS = int(os.getenv("MICRO_RB_BASE_UNITS", "20000"))

CAP_MIN = float(os.getenv("MICRO_RB_CAP_MIN", "0.15"))
CAP_MAX = float(os.getenv("MICRO_RB_CAP_MAX", "0.95"))

MAX_FACTOR_AGE_SEC = float(os.getenv("MICRO_RB_MAX_FACTOR_AGE_SEC", "90.0"))
RANGE_ONLY_SCORE = float(os.getenv("MICRO_RB_RANGE_ONLY_SCORE", "0.45"))

# Size boost for strong range setups
SIZE_MULT_BASE = float(os.getenv("MICRO_RB_SIZE_MULT_BASE", "1.07"))
SIZE_MULT_SCORE_START = float(os.getenv("MICRO_RB_SIZE_MULT_SCORE_START", "0.45"))
SIZE_MULT_SLOPE = float(os.getenv("MICRO_RB_SIZE_MULT_SLOPE", "0.5"))
SIZE_MULT_MIN = float(os.getenv("MICRO_RB_SIZE_MULT_MIN", "0.9"))
SIZE_MULT_MAX = float(os.getenv("MICRO_RB_SIZE_MULT_MAX", "1.2"))
SIZE_MULT_MIN_FMR = float(os.getenv("MICRO_RB_SIZE_MULT_MIN_FMR", "0.12"))
