"""Per-trade EXIT loop for M1Scalper (scalp pocket)."""

from __future__ import annotations

import asyncio
import logging

from workers.common.simple_exit import SimpleExitWorker

LOG = logging.getLogger(__name__)

ALLOWED_TAGS = {"M1Scalper", "m1scalper", "m1_scalper"}


async def m1_scalper_exit_worker() -> None:
    worker = SimpleExitWorker(
        pocket="scalp",
        strategy_tags=ALLOWED_TAGS,
        profit_take=2.0,
        trail_start=2.6,
        trail_backoff=0.9,
        stop_loss=1.6,
        max_hold_sec=15 * 60,
        loop_interval=0.8,
        lock_buffer=0.5,
        use_entry_meta=True,
        structure_break=True,
    )
    await worker.run()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(m1_scalper_exit_worker())
