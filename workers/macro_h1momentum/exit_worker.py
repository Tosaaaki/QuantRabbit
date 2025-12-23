"""Exit loop for H1Momentum macro worker."""

from __future__ import annotations

import asyncio
import logging

from workers.common.simple_exit import SimpleExitWorker

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"H1Momentum"}


async def h1momentum_exit_worker() -> None:
    worker = SimpleExitWorker(
        pocket="macro",
        strategy_tags=ALLOWED_TAGS,
        profit_take=5.0,
        trail_start=6.5,
        trail_backoff=2.0,
        stop_loss=3.0,
        max_hold_sec=3 * 3600,
        loop_interval=2.0,
        lock_buffer=0.9,
    )
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(h1momentum_exit_worker())
