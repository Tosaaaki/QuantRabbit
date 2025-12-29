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
            for h in range(start, end + 1):
                if 0 <= h <= 23:
                    hours.add(h)
            continue
        try:
            h = int(float(token))
        except ValueError:
            continue
        if 0 <= h <= 23:
            hours.add(h)
    return hours

POCKET = "micro"
LOOP_INTERVAL_SEC = float(os.getenv("MICRO_MULTI_LOOP_INTERVAL_SEC", "8.0"))
ENABLED = os.getenv("MICRO_MULTI_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[MicroMulti]"

CONFIDENCE_FLOOR = 35
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("MICRO_MULTI_MIN_UNITS", "1500"))
BASE_ENTRY_UNITS = int(os.getenv("MICRO_MULTI_BASE_UNITS", "12000"))
MAX_MARGIN_USAGE = float(os.getenv("MICRO_MULTI_MAX_MARGIN_USAGE", "0.9"))

CAP_MIN = float(os.getenv("MICRO_MULTI_CAP_MIN", "0.15"))
CAP_MAX = float(os.getenv("MICRO_MULTI_CAP_MAX", "0.8"))

# TrendMomentumMicro が沈んだ時間帯（UTC 10/11）をデフォルトブロック
TREND_BLOCK_HOURS_UTC = frozenset(
    _parse_hours(os.getenv("MICRO_TREND_BLOCK_HOURS_UTC", "10,11"))
)
