"""Exit loop for impulse_break_s5 worker (scalp pocket)."""

from __future__ import annotations

import asyncio
import logging

from workers.common.simple_exit import SimpleExitWorker

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"impulse_break_s5"}


async def impulse_break_s5_exit_worker() -> None:
    worker = SimpleExitWorker(
        pocket="scalp",
        strategy_tags=ALLOWED_TAGS,
        profit_take=2.2,
        trail_start=3.0,
        trail_backoff=1.0,
        stop_loss=1.6,
        max_hold_sec=25 * 60,
        loop_interval=1.0,
        lock_buffer=0.6,
        use_entry_meta=True,
        structure_break=True,
    )
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(impulse_break_s5_exit_worker())
