from __future__ import annotations

import os

POCKET = "micro"
LOOP_INTERVAL_SEC = float(os.getenv("BBRSI_LOOP_INTERVAL_SEC", "8.0"))
ENABLED = os.getenv("BBRSI_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[BB_RSI]"

CONFIDENCE_FLOOR = 35
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("BBRSI_MIN_UNITS", "1500"))
BASE_ENTRY_UNITS = int(os.getenv("BBRSI_BASE_UNITS", "12000"))
MAX_MARGIN_USAGE = float(os.getenv("BBRSI_MAX_MARGIN_USAGE", "0.9"))

CAP_MIN = float(os.getenv("BBRSI_CAP_MIN", "0.15"))
CAP_MAX = float(os.getenv("BBRSI_CAP_MAX", "0.95"))
