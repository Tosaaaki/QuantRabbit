from __future__ import annotations

import os

ENV_PREFIX = "TRRECLAIM"
POCKET = "scalp"
PIP_VALUE = 0.01
STRATEGY_TAG = "TrendReclaimLong"
LOOP_INTERVAL_SEC = float(os.getenv("TRRECLAIM_LOOP_INTERVAL_SEC", "5.0"))
ENABLED = os.getenv("TRRECLAIM_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[TrendReclaimLong]"

CONFIDENCE_FLOOR = int(float(os.getenv("TRRECLAIM_CONF_FLOOR", "52")))
CONFIDENCE_CEIL = int(float(os.getenv("TRRECLAIM_CONF_CEIL", "92")))
MIN_ENTRY_CONF = int(float(os.getenv("TRRECLAIM_MIN_ENTRY_CONF", str(CONFIDENCE_FLOOR))))

MIN_UNITS = int(os.getenv("TRRECLAIM_MIN_UNITS", "1000"))
BASE_ENTRY_UNITS = int(os.getenv("TRRECLAIM_BASE_UNITS", "7000"))
MAX_SPREAD_PIPS = float(os.getenv("TRRECLAIM_MAX_SPREAD_PIPS", "1.2"))
COOLDOWN_SEC = float(os.getenv("TRRECLAIM_COOLDOWN_SEC", "45.0"))
BLOCK_IF_LONG_OPEN = os.getenv("TRRECLAIM_BLOCK_IF_LONG_OPEN", "0").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}

CAP_MIN = float(os.getenv("TRRECLAIM_CAP_MIN", "0.10"))
CAP_MAX = float(os.getenv("TRRECLAIM_CAP_MAX", "0.85"))
RANGE_BLOCK_SCORE = float(os.getenv("TRRECLAIM_RANGE_BLOCK_SCORE", "0.58"))

DYN_ALLOC_ENABLED = os.getenv("TRRECLAIM_DYN_ALLOC_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
DYN_ALLOC_PATH = os.getenv("TRRECLAIM_DYN_ALLOC_PATH", "config/dynamic_alloc.json")
DYN_ALLOC_TTL_SEC = float(os.getenv("TRRECLAIM_DYN_ALLOC_TTL_SEC", "20"))
DYN_ALLOC_MIN_TRADES = int(os.getenv("TRRECLAIM_DYN_ALLOC_MIN_TRADES", "8"))
DYN_ALLOC_LOSER_SCORE = float(os.getenv("TRRECLAIM_DYN_ALLOC_LOSER_SCORE", "0.30"))
DYN_ALLOC_LOSER_BLOCK = os.getenv("TRRECLAIM_DYN_ALLOC_LOSER_BLOCK", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
DYN_ALLOC_MULT_MIN = float(os.getenv("TRRECLAIM_DYN_ALLOC_MULT_MIN", "0.7"))
DYN_ALLOC_MULT_MAX = float(os.getenv("TRRECLAIM_DYN_ALLOC_MULT_MAX", "1.8"))

EXIT_LOOP_INTERVAL_SEC = float(os.getenv("TRRECLAIM_EXIT_LOOP_INTERVAL_SEC", "0.9"))
EXIT_MIN_HOLD_SEC = float(os.getenv("TRRECLAIM_EXIT_MIN_HOLD_SEC", "20.0"))
EXIT_MAX_HOLD_SEC = float(os.getenv("TRRECLAIM_EXIT_MAX_HOLD_SEC", "900.0"))
EXIT_PROFIT_PIPS = float(os.getenv("TRRECLAIM_EXIT_PROFIT_PIPS", "4.5"))
EXIT_TRAIL_START_PIPS = float(os.getenv("TRRECLAIM_EXIT_TRAIL_START_PIPS", "6.0"))
EXIT_TRAIL_BACKOFF_PIPS = float(os.getenv("TRRECLAIM_EXIT_TRAIL_BACKOFF_PIPS", "2.0"))
EXIT_LOCK_BUFFER_PIPS = float(os.getenv("TRRECLAIM_EXIT_LOCK_BUFFER_PIPS", "1.6"))
EXIT_LOCK_TRIGGER_PIPS = float(os.getenv("TRRECLAIM_EXIT_LOCK_TRIGGER_PIPS", "3.2"))
EXIT_REVERSAL_WINDOW_SEC = float(os.getenv("TRRECLAIM_EXIT_REV_WINDOW_SEC", "30.0"))
EXIT_REVERSAL_PIPS = float(os.getenv("TRRECLAIM_EXIT_REV_PIPS", "1.8"))
EXIT_ATR_SPIKE_PIPS = float(os.getenv("TRRECLAIM_EXIT_ATR_SPIKE_PIPS", "7.0"))
EXIT_HARD_STOP_PIPS = float(os.getenv("TRRECLAIM_EXIT_HARD_STOP_PIPS", "6.0"))
EXIT_TP_HINT_RATIO = float(os.getenv("TRRECLAIM_EXIT_TP_HINT_RATIO", "0.85"))
