"""
news_fetcher (stub)
~~~~~~~~~~~~~~~~~~~
ニュース連動機能は廃止済みのため、何もせず即座に戻るスタブを提供する。
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Dict

LOG = logging.getLogger(__name__)


async def fetch_loop(interval_sec: float = 300.0) -> None:
    """
    Legacy entrypoint kept for compatibility; simply idles.
    """
    LOG.info("news_fetcher disabled; idle loop running")
    while True:
        await asyncio.sleep(interval_sec)


async def iter_news() -> AsyncIterator[Dict]:
    """
    Yield nothing; helper for callers that expect an async iterator.
    """
    if False:  # pragma: no cover - intentional no-op
        yield {}

