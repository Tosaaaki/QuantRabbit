from __future__ import annotations

import os

from utils.env_utils import env_bool

ENV_PREFIX = "HAR"
POCKET = "micro"

LOOP_INTERVAL_SEC = float(os.getenv("HAR_LOOP_INTERVAL_SEC", "5.0"))
ENABLED = os.getenv("HAR_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[MicroAdaptiveRevert]"

CONFIDENCE_FLOOR = int(float(os.getenv("HAR_CONF_FLOOR", "42")))
CONFIDENCE_CEIL = int(float(os.getenv("HAR_CONF_CEIL", "89")))
MIN_ENTRY_CONF = int(float(os.getenv("HAR_MIN_ENTRY_CONF", str(CONFIDENCE_FLOOR))))

MIN_UNITS = int(os.getenv("HAR_MIN_UNITS", "1100"))
BASE_ENTRY_UNITS = int(os.getenv("HAR_BASE_UNITS", "20000"))

CAP_MIN = float(os.getenv("HAR_CAP_MIN", "0.18"))
CAP_MAX = float(os.getenv("HAR_CAP_MAX", "0.94"))

RANGE_SCORE_MIN = float(os.getenv("HAR_RANGE_SCORE_MIN", "0.34"))
MAX_FACTOR_AGE_SEC = float(os.getenv("HAR_MAX_FACTOR_AGE_SEC", "95.0"))
PATTERN_GATE_OPT_IN = env_bool("HAR_PATTERN_GATE_OPT_IN", True, prefix=ENV_PREFIX)
