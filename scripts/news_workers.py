#!/usr/bin/env python3
"""Utility runner for news fetch + summary ingest loops."""
from __future__ import annotations

import asyncio
import logging

from analysis.summary_ingestor import ingest_loop as summary_ingest_loop
from market_data.news_fetcher import fetch_loop as news_fetch_loop


async def _run() -> None:
    tasks = [
        asyncio.create_task(news_fetch_loop(), name="news_fetch_loop"),
        asyncio.create_task(summary_ingest_loop(), name="summary_ingest_loop"),
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logging.info("news workers stopped by user")


if __name__ == "__main__":
    main()
