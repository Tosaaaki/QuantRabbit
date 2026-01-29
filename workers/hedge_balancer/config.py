import os

ENABLED = os.getenv("HEDGE_BALANCER_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "off",
    "no",
}
LOOP_INTERVAL_SEC = float(os.getenv("HEDGE_BALANCER_INTERVAL_SEC", "7.0") or 7.0)
TRIGGER_MARGIN_USAGE = float(os.getenv("HEDGE_TRIGGER_MARGIN_USAGE", "0.88") or 0.88)
TARGET_MARGIN_USAGE = float(os.getenv("HEDGE_TARGET_MARGIN_USAGE", "0.82") or 0.82)
TRIGGER_FREE_MARGIN_RATIO = float(os.getenv("HEDGE_TRIGGER_FREE_MARGIN_RATIO", "0.08") or 0.08)
MIN_NET_UNITS = int(float(os.getenv("HEDGE_MIN_NET_UNITS", "15000") or 15000))
MIN_HEDGE_UNITS = int(float(os.getenv("HEDGE_MIN_HEDGE_UNITS", "10000") or 10000))
MAX_HEDGE_UNITS = int(float(os.getenv("HEDGE_MAX_HEDGE_UNITS", "90000") or 90000))
MAX_REDUCTION_FRACTION = float(os.getenv("HEDGE_MAX_REDUCTION_FRACTION", "0.55") or 0.55)
COOLDOWN_SEC = float(os.getenv("HEDGE_COOLDOWN_SEC", "20.0") or 20.0)
POCKET = os.getenv("HEDGE_POCKET", "macro").strip() or "macro"
CONFIDENCE = int(os.getenv("HEDGE_CONFIDENCE", "90") or 90)
SL_PIPS = float(os.getenv("HEDGE_SL_PIPS", "5.0") or 5.0)
TP_PIPS = float(os.getenv("HEDGE_TP_PIPS", "5.0") or 5.0)
MIN_PRICE = float(os.getenv("HEDGE_MIN_PRICE", "90.0") or 90.0)
LOG_PREFIX = "[HEDGE]"

# Hedge lock unwind (both-side positions with small net exposure)
LOCK_ENABLED = os.getenv("HEDGE_LOCK_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "off",
    "no",
}
LOCK_NET_MAX_UNITS = int(float(os.getenv("HEDGE_LOCK_NET_MAX_UNITS", "4000") or 4000))
LOCK_GROSS_MIN_UNITS = int(float(os.getenv("HEDGE_LOCK_GROSS_MIN_UNITS", "18000") or 18000))
LOCK_MIN_SIDE_UNITS = int(float(os.getenv("HEDGE_LOCK_MIN_SIDE_UNITS", "6000") or 6000))
LOCK_MIN_UNITS = int(float(os.getenv("HEDGE_LOCK_MIN_UNITS", "3000") or 3000))
LOCK_MAX_UNITS = int(float(os.getenv("HEDGE_LOCK_MAX_UNITS", "18000") or 18000))
LOCK_MAX_REDUCTION_FRACTION = float(os.getenv("HEDGE_LOCK_MAX_REDUCTION_FRACTION", "0.25") or 0.25)
LOCK_COOLDOWN_SEC = float(os.getenv("HEDGE_LOCK_COOLDOWN_SEC", "45.0") or 45.0)
LOCK_SCORE_GAP = float(os.getenv("HEDGE_LOCK_SCORE_GAP", "0.35") or 0.35)
LOCK_SCORE_MIN = float(os.getenv("HEDGE_LOCK_SCORE_MIN", "0.2") or 0.2)
LOCK_ALLOW_RANGE = os.getenv("HEDGE_LOCK_ALLOW_RANGE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
