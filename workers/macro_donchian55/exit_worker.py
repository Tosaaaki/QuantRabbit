"""Exit loop for Donchian55 macro worker."""

from __future__ import annotations

import asyncio
import logging

from workers.common.simple_exit import SimpleExitWorker

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"Donchian55"}


async def donchian55_exit_worker() -> None:
    worker = SimpleExitWorker(
        pocket="macro",
        strategy_tags=ALLOWED_TAGS,
        profit_take=7.0,
        trail_start=9.0,
        trail_backoff=3.0,
        stop_loss=4.0,
        max_hold_sec=6 * 3600,
        loop_interval=2.0,
        lock_buffer=1.2,
    )
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(donchian55_exit_worker())
