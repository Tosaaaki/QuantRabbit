"""Exit loop for onepip_maker_s1 worker (scalp pocket)."""

from __future__ import annotations

import asyncio
import logging

from workers.common.simple_exit import SimpleExitWorker

LOG = logging.getLogger(__name__)


async def onepip_maker_s1_exit_worker() -> None:
    worker = SimpleExitWorker(
        pocket="scalp",
        strategy_tags=set(),  # target all trades in pocket (strategy tag not set)
        profit_take=1.2,
        trail_start=1.6,
        trail_backoff=0.5,
        stop_loss=1.0,
        max_hold_sec=12 * 60,
        loop_interval=0.6,
        lock_buffer=0.3,
    )
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(onepip_maker_s1_exit_worker())
