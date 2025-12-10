"""
Stub implementation for the legacy news ingestor.

ニュース配信を廃止したため、GCS への取得や DB 挿入は行わず、
常に空のニュースとイベント無しを返す。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict

LOG = logging.getLogger(__name__)


async def ingest_loop(interval_sec: int = 30) -> None:
    """
    No-op loop kept for compatibility with existing service wiring.
    """
    LOG.info("news ingest loop disabled (news pipeline removed)")
    while True:
        await asyncio.sleep(interval_sec)


def get_latest_news(limit_short: int = 2, limit_long: int = 2) -> Dict[str, list]:
    """
    Return an empty news cache for callers that expect historical summaries.
    """
    return {"short": [], "long": []}


def check_event_soon(within_minutes: int = 30, min_impact: int = 3) -> bool:
    """
    Always return False because news/event gating has been removed.
    """
    return False

