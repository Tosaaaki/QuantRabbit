from __future__ import annotations

import asyncio
from typing import Dict, List


async def ingest_loop(interval_sec: int = 30):
    # no-op for offline replay
    while True:
        await asyncio.sleep(interval_sec)


def get_latest_news(limit_short: int = 2, limit_long: int = 2) -> dict:
    # return empty news cache for offline replay
    return {"short": [], "long": []}


def check_event_soon(within_minutes: int = 30, min_impact: int = 3) -> bool:
    return False

