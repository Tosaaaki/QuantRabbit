"""Dedicated exit worker for MomentumPulse strategy.

Thin launcher: sets strategy-specific env vars, then delegates to the
shared micro_runtime exit loop to avoid code duplication across micro workers.
"""
from __future__ import annotations

import asyncio
import logging
import os

os.environ.setdefault("MICRO_MULTI_EXIT_ENABLED", "1")
os.environ.setdefault("MICRO_MULTI_LOG_PREFIX", "[MomentumPulse]")
os.environ.setdefault("MICRO_MULTI_EXIT_TAG_ALLOWLIST", "MomentumPulse")

from workers.micro_runtime.exit_worker import micro_runtime_exit_worker

LOG = logging.getLogger(__name__)


async def micro_momentumpulse_exit_worker() -> None:
    await micro_runtime_exit_worker()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(micro_momentumpulse_exit_worker())
