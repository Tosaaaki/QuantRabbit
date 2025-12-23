"""Exit loop for micro multi-strategy worker."""

from __future__ import annotations

import asyncio
import logging

from workers.common.simple_exit import SimpleExitWorker

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {
    "MomentumBurst",
    "MicroMomentumStack",
    "MicroPullbackEMA",
    "MicroLevelReactor",
    "MicroRangeBreak",
    "MicroVWAPBound",
    "TrendMomentumMicro",
}


async def micro_multistrat_exit_worker() -> None:
    worker = SimpleExitWorker(
        pocket="micro",
        strategy_tags=ALLOWED_TAGS,
        profit_take=2.0,
        trail_start=2.8,
        trail_backoff=0.9,
        stop_loss=1.4,
        max_hold_sec=45 * 60,
        loop_interval=1.0,
        lock_buffer=0.4,
    )
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(micro_multistrat_exit_worker())
