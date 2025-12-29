from __future__ import annotations

import os

POCKET = "scalp"
LOOP_INTERVAL_SEC = float(os.getenv("SCALP_MULTI_LOOP_INTERVAL_SEC", "6.0"))
ENABLED = os.getenv("SCALP_MULTI_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[ScalpMulti]"

CONFIDENCE_FLOOR = 30
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("SCALP_MULTI_MIN_UNITS", "1000"))
BASE_ENTRY_UNITS = int(os.getenv("SCALP_MULTI_BASE_UNITS", "8000"))
MAX_MARGIN_USAGE = float(os.getenv("SCALP_MULTI_MAX_MARGIN_USAGE", "0.85"))

CAP_MIN = float(os.getenv("SCALP_MULTI_CAP_MIN", "0.1"))
CAP_MAX = float(os.getenv("SCALP_MULTI_CAP_MAX", "0.65"))

