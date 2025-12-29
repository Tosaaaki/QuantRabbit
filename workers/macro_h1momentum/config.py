from __future__ import annotations

import os

POCKET = "macro"
LOOP_INTERVAL_SEC = float(os.getenv("H1M_LOOP_INTERVAL_SEC", "15.0"))
ENABLED = os.getenv("H1M_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[H1Momentum]"

# Risk / sizing
CONFIDENCE_FLOOR = 40
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("H1M_MIN_UNITS", "2000"))
BASE_ENTRY_UNITS = int(os.getenv("H1M_BASE_UNITS", "40000"))  # default 0.4 lot
MAX_MARGIN_USAGE = float(os.getenv("H1M_MAX_MARGIN_USAGE", "0.92"))

# Dynamic cap clamp
CAP_MIN = float(os.getenv("H1M_CAP_MIN", "0.25"))
CAP_MAX = float(os.getenv("H1M_CAP_MAX", "0.95"))


