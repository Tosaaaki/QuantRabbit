from __future__ import annotations

import os


def _parse_hours(raw: str) -> set[int]:
    hours: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            try:
                start = int(float(start_s))
                end = int(float(end_s))
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for hour in range(start, end + 1):
                if 0 <= hour <= 23:
                    hours.add(hour)
            continue
        try:
            hour_val = int(float(token))
        except ValueError:
            continue
        if 0 <= hour_val <= 23:
            hours.add(hour_val)
    return hours


POCKET = "micro"
LOOP_INTERVAL_SEC = float(os.getenv("MSTACK_LOOP_INTERVAL_SEC", "8.0"))
ENABLED = os.getenv("MSTACK_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[MicroMomentumStack]"

CONFIDENCE_FLOOR = 35
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("MSTACK_MIN_UNITS", "1500"))
BASE_ENTRY_UNITS = int(os.getenv("MSTACK_BASE_UNITS", "20000"))

CAP_MIN = float(os.getenv("MSTACK_CAP_MIN", "0.15"))
CAP_MAX = float(os.getenv("MSTACK_CAP_MAX", "0.95"))

MAX_FACTOR_AGE_SEC = float(os.getenv("MSTACK_MAX_FACTOR_AGE_SEC", "90.0"))
RANGE_ONLY_SCORE = float(os.getenv("MSTACK_RANGE_ONLY_SCORE", "0.45"))

_RAW_BLOCK = os.getenv("MSTACK_BLOCK_HOURS_UTC")
if _RAW_BLOCK is None:
    _RAW_BLOCK = os.getenv("MICRO_TREND_BLOCK_HOURS_UTC", "")
TREND_BLOCK_HOURS_UTC = frozenset(_parse_hours(_RAW_BLOCK or ""))
