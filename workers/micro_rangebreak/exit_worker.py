"""Dedicated exit worker for MicroRangeBreak strategy."""

from __future__ import annotations

import asyncio
import logging
import os

os.environ["MICRO_MULTI_EXIT_ENABLED"] = "1"
os.environ["MICRO_MULTI_LOG_PREFIX"] = "[MicroRangeBreak]"
os.environ["MICRO_MULTI_EXIT_TAG_ALLOWLIST"] = "MicroRangeBreak"

from workers.micro_multistrat.exit_worker import micro_multistrat_exit_worker

LOG = logging.getLogger(__name__)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(micro_multistrat_exit_worker())
