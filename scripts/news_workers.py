#!/usr/bin/env python3
"""Deprecated stub: news pipeline has been removed."""
from __future__ import annotations

import asyncio
import logging


async def _idle() -> None:
    logging.info("news pipeline removed; news_workers is now a no-op")
    while True:
        await asyncio.sleep(3600)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_idle())
    except KeyboardInterrupt:
        logging.info("news worker stub stopped by user")


if __name__ == "__main__":
    main()
