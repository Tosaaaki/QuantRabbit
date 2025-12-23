"""Exit loop for mirror_spike worker (scalp pocket)."""

from __future__ import annotations

import asyncio
import logging

from workers.common.simple_exit import SimpleExitWorker

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"mirror_spike"}


async def mirror_spike_exit_worker() -> None:
    worker = SimpleExitWorker(
        pocket="scalp",
        strategy_tags=ALLOWED_TAGS,
        profit_take=1.8,
        trail_start=2.4,
        trail_backoff=0.8,
        stop_loss=1.2,
        max_hold_sec=18 * 60,
        loop_interval=0.8,
        lock_buffer=0.45,
    )
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(mirror_spike_exit_worker())
