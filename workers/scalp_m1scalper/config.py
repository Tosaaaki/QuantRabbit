from __future__ import annotations

import os

POCKET = "scalp"
LOOP_INTERVAL_SEC = float(os.getenv("M1SCALP_LOOP_INTERVAL_SEC", "6.0"))
ENABLED = os.getenv("M1SCALP_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[M1Scalper]"

CONFIDENCE_FLOOR = 30
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("M1SCALP_MIN_UNITS", "2000"))
BASE_ENTRY_UNITS = int(os.getenv("M1SCALP_BASE_UNITS", "2000"))
MAX_MARGIN_USAGE = float(os.getenv("M1SCALP_MAX_MARGIN_USAGE", "0.82"))

CAP_MIN = float(os.getenv("M1SCALP_CAP_MIN", "0.1"))
CAP_MAX = float(os.getenv("M1SCALP_CAP_MAX", "0.25"))
# クールダウン/同時建玉制御（デフォルトを明示）
COOLDOWN_SEC = float(os.getenv("M1SCALP_COOLDOWN_SEC", "120"))
MAX_OPEN_TRADES = int(os.getenv("M1SCALP_MAX_OPEN_TRADES", "2"))
