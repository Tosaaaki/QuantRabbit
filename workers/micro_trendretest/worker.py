"""Legacy compatibility entrypoint for ``workers.micro_trendretest``.

Historically, some deployments still reference ``workers.micro_trendretest`` as a
standalone ENTRY worker. The implementation now routes that legacy name through the
micro-multi dispatcher while keeping the behavior scoped to TrendRetest-only.
"""

from __future__ import annotations

import asyncio
import os

# Keep behavior aligned with existing micro-multi policy: only run TrendRetest in
# this legacy entrypoint unless explicitly overridden by environment.
os.environ.setdefault("MICRO_STRATEGY_ALLOWLIST", "MicroTrendRetest")

from workers.micro_multistrat.worker import micro_multi_worker


async def micro_trendretest_worker() -> None:
    await micro_multi_worker()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(micro_trendretest_worker())

