from __future__ import annotations

import os

POCKET = "micro"
LOOP_INTERVAL_SEC = float(os.getenv("RRL_LOOP_INTERVAL_SEC", "6.0"))
ENABLED = os.getenv("RRL_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[RANGE_REVERT_LITE]"

CONFIDENCE_FLOOR = int(float(os.getenv("RRL_CONF_FLOOR", "40")))
CONFIDENCE_CEIL = int(float(os.getenv("RRL_CONF_CEIL", "88")))
MIN_ENTRY_CONF = int(float(os.getenv("RRL_MIN_ENTRY_CONF", str(CONFIDENCE_FLOOR))))

MIN_UNITS = int(os.getenv("RRL_MIN_UNITS", "1200"))
BASE_ENTRY_UNITS = int(os.getenv("RRL_BASE_UNITS", "18000"))
MAX_MARGIN_USAGE = float(os.getenv("RRL_MAX_MARGIN_USAGE", "0.9"))

CAP_MIN = float(os.getenv("RRL_CAP_MIN", "0.20"))
CAP_MAX = float(os.getenv("RRL_CAP_MAX", "0.95"))

RANGE_SCORE_MIN = float(os.getenv("RRL_RANGE_SCORE_MIN", "0.40"))
