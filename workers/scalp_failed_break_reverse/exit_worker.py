"""FailedBreakReverse exit worker (tag-scoped wrapper over M1Scalper exit)."""

from __future__ import annotations

import asyncio
import logging
import os

os.environ.setdefault("M1SCALP_EXIT_TAG_ALLOWLIST", "FailedBreakReverse")

from workers.scalp_m1scalper import exit_worker as _base_exit


async def failed_break_reverse_exit_worker() -> None:
    await _base_exit.m1_scalper_exit_worker()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(failed_break_reverse_exit_worker())

