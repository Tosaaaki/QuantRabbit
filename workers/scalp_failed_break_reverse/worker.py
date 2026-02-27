"""FailedBreakReverse entry worker (dedicated wrapper over M1Scalper engine)."""

from __future__ import annotations

import asyncio
import logging
import os

os.environ.setdefault("M1SCALP_ENABLED", "1")
os.environ.setdefault("M1SCALP_SIGNAL_TAG_CONTAINS", "vshape-rebound")
os.environ.setdefault("M1SCALP_STRATEGY_TAG_OVERRIDE", "FailedBreakReverse")
os.environ.setdefault("M1SCALP_ALLOW_REVERSION", "1")
os.environ.setdefault("M1SCALP_ALLOW_TREND", "0")

from workers.scalp_m1scalper import worker as _base_worker


async def failed_break_reverse_worker() -> None:
    await _base_worker.scalp_m1_worker()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(failed_break_reverse_worker())

