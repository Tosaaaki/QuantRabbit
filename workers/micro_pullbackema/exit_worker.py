"""Dedicated exit worker for MicroPullbackEMA strategy."""

from __future__ import annotations

import asyncio
import logging
import os

os.environ["MICRO_MULTI_EXIT_ENABLED"] = "1"
os.environ["MICRO_MULTI_LOG_PREFIX"] = "[MicroPullbackEMA]"
os.environ["MICRO_MULTI_EXIT_TAG_ALLOWLIST"] = "MicroPullbackEMA"

from workers.micro_runtime.exit_worker import micro_multistrat_exit_worker

LOG = logging.getLogger(__name__)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(micro_multistrat_exit_worker())

