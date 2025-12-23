"""Exit loop for scalp multi-strategy worker."""

from __future__ import annotations

import asyncio
import logging

from workers.common.simple_exit import SimpleExitWorker

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"RangeFader", "PulseBreak", "ImpulseRetraceScalp"}


async def scalp_multistrat_exit_worker() -> None:
    worker = SimpleExitWorker(
        pocket="scalp",
        strategy_tags=ALLOWED_TAGS,
        profit_take=1.6,
        trail_start=2.2,
        trail_backoff=0.8,
        stop_loss=1.0,
        max_hold_sec=12 * 60,
        loop_interval=0.7,
        lock_buffer=0.35,
    )
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(scalp_multistrat_exit_worker())
