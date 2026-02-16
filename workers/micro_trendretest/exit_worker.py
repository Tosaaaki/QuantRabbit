"""Legacy compatibility EXIT worker for ``workers.micro_trendretest``.

Routes old module entrypoints to the shared micro-multi EXIT worker while
keeping tag filtering aligned to TrendRetest only.
"""

from __future__ import annotations

import asyncio

from workers.micro_multistrat import exit_worker as _exit_worker


def _configure_tag_filter() -> None:
    _exit_worker.ALLOWED_TAGS.clear()
    _exit_worker.ALLOWED_TAGS.update({"MicroTrendRetest"})


async def micro_trendretest_exit_worker() -> None:
    _configure_tag_filter()
    await _exit_worker.micro_multistrat_exit_worker()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(micro_trendretest_exit_worker())

