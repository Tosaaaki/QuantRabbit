from __future__ import annotations

import os


def _parse_hours(raw: str) -> set[int]:
    hours: set[int] = set()
    for token in str(raw or "").split(","):
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


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


POCKET = "micro"
INSTRUMENT = (os.getenv("MICRO_TRT_INSTRUMENT", "USD_JPY") or "USD_JPY").strip() or "USD_JPY"

SIGNAL_TF = (os.getenv("MICRO_TRT_SIGNAL_TF", "M5") or "M5").strip().upper()
if SIGNAL_TF not in {"M1", "M5"}:
    SIGNAL_TF = "M5"

LOOP_INTERVAL_SEC = float(os.getenv("MICRO_TRT_LOOP_INTERVAL_SEC", "8.0"))
ENABLED = _bool("MICRO_TRT_ENABLED", False)
DRY_RUN = _bool("MICRO_TRT_DRY_RUN", False)
LOG_PREFIX = "[MicroTrendRetest]"

CONFIDENCE_FLOOR = int(os.getenv("MICRO_TRT_CONF_FLOOR", "55"))
CONFIDENCE_CEIL = int(os.getenv("MICRO_TRT_CONF_CEIL", "90"))

MIN_UNITS = int(os.getenv("MICRO_TRT_MIN_UNITS", "1500"))
BASE_ENTRY_UNITS = int(os.getenv("MICRO_TRT_BASE_UNITS", "18000"))

MAX_FACTOR_AGE_SEC = float(os.getenv("MICRO_TRT_MAX_FACTOR_AGE_SEC", "120.0"))
MAX_TICK_LAG_MS = float(os.getenv("MICRO_TRT_MAX_TICK_LAG_MS", "6000.0"))
MAX_SPREAD_PIPS = float(os.getenv("MICRO_TRT_MAX_SPREAD_PIPS", "2.2"))

MIN_ENTRY_INTERVAL_SEC = float(os.getenv("MICRO_TRT_MIN_ENTRY_INTERVAL_SEC", "20.0"))

BLOCK_HOURS_UTC = frozenset(_parse_hours(os.getenv("MICRO_TRT_BLOCK_HOURS_UTC", "")))

