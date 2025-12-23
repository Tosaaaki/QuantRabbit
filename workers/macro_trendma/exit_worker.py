"""Exit loop for TrendMA macro worker."""

from __future__ import annotations

import asyncio
import logging

from workers.common.simple_exit import SimpleExitWorker

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"TrendMA"}


async def trendma_exit_worker() -> None:
    worker = SimpleExitWorker(
        pocket="macro",
        strategy_tags=ALLOWED_TAGS,
        profit_take=6.0,
        trail_start=7.0,
        trail_backoff=2.5,
        stop_loss=3.5,
        max_hold_sec=4 * 3600,
        loop_interval=2.0,
        lock_buffer=1.0,
    )
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(trendma_exit_worker())
