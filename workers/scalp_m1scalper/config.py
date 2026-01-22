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

POCKET = "scalp"
LOOP_INTERVAL_SEC = float(os.getenv("M1SCALP_LOOP_INTERVAL_SEC", "6.0"))
ENABLED = os.getenv("M1SCALP_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[M1Scalper]"

CONFIDENCE_FLOOR = 30
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("M1SCALP_MIN_UNITS", "1000"))
BASE_ENTRY_UNITS = int(os.getenv("M1SCALP_BASE_UNITS", "6000"))
MAX_MARGIN_USAGE = float(os.getenv("M1SCALP_MAX_MARGIN_USAGE", "0.9"))

CAP_MIN = float(os.getenv("M1SCALP_CAP_MIN", "0.1"))
CAP_MAX = float(os.getenv("M1SCALP_CAP_MAX", "0.9"))
# クールダウン/同時建玉制御（デフォルトを明示）
COOLDOWN_SEC = float(os.getenv("M1SCALP_COOLDOWN_SEC", "120"))
MAX_OPEN_TRADES = int(os.getenv("M1SCALP_MAX_OPEN_TRADES", "2"))
# 直近の大敗時間帯をデフォルトでブロック（00/10/20/23 UTC）
BLOCK_HOURS_UTC = frozenset(
    _parse_hours(os.getenv("M1SCALP_BLOCK_HOURS_UTC", ""))
)
