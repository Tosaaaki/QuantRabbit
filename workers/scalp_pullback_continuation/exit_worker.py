"""PullbackContinuation exit worker (tag-scoped wrapper over M1Scalper exit)."""

from __future__ import annotations

import asyncio
import logging
import os

os.environ.setdefault("M1SCALP_EXIT_TAG_ALLOWLIST", "PullbackContinuation")

from workers.scalp_m1scalper import exit_worker as _base_exit


async def pullback_continuation_exit_worker() -> None:
    await _base_exit.m1_scalper_exit_worker()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(pullback_continuation_exit_worker())

