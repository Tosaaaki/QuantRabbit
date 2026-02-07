from __future__ import annotations

import os

ENV_PREFIX = "MOMB"
POCKET = "micro"
LOOP_INTERVAL_SEC = float(os.getenv("MOMB_LOOP_INTERVAL_SEC", "8.0"))
ENABLED = os.getenv("MOMB_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[MomentumBurst]"

CONFIDENCE_FLOOR = 35
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("MOMB_MIN_UNITS", "1500"))
BASE_ENTRY_UNITS = int(os.getenv("MOMB_BASE_UNITS", "20000"))

CAP_MIN = float(os.getenv("MOMB_CAP_MIN", "0.15"))
CAP_MAX = float(os.getenv("MOMB_CAP_MAX", "0.95"))

MAX_FACTOR_AGE_SEC = float(os.getenv("MOMB_MAX_FACTOR_AGE_SEC", "90.0"))
RANGE_ONLY_SCORE = float(os.getenv("MOMB_RANGE_ONLY_SCORE", "0.45"))
