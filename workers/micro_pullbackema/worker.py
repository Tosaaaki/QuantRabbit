"""Dedicated entry worker for MicroPullbackEMA strategy."""

from __future__ import annotations

import asyncio
import logging
import os

os.environ["MICRO_STRATEGY_ALLOWLIST"] = "MicroPullbackEMA"
os.environ["MICRO_MULTI_LOG_PREFIX"] = "[MicroPullbackEMA]"

from workers.micro_runtime import worker as _base_worker

LOG = logging.getLogger(__name__)


async def micro_pullbackema_worker() -> None:
    await _base_worker.micro_multi_worker()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(micro_pullbackema_worker())

