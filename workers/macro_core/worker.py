"""Macro pocket plan executor worker."""

from __future__ import annotations

import asyncio
import logging

from analysis.plan_bus import PlanCursor
from workers.common.core_executor import PocketPlanExecutor

from . import config

LOG = logging.getLogger(__name__)


async def macro_core_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled by configuration", config.LOG_PREFIX)
        return

    cursor = PlanCursor()
    executor = PocketPlanExecutor("macro", log_prefix=config.LOG_PREFIX)
    LOG.info("%s worker starting (interval %.2fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            plan = cursor.consume("macro")
            if not plan:
                continue
            if plan.is_stale(config.PLAN_STALE_SEC):
                LOG.debug(
                    "%s plan stale age=%.2fs", config.LOG_PREFIX, plan.age_seconds()
                )
                continue
            try:
                await executor.process_plan(plan)
            except Exception as exc:  # pragma: no cover - defensive
                LOG.exception("%s plan processing failed: %s", config.LOG_PREFIX, exc)
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        executor.close()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    LOG.info("%s worker starting", config.LOG_PREFIX)
    asyncio.run(macro_core_worker())
