"""Experimental 5-second ping scalp exit worker (D variant wrapper).

D variant reuses the C exit implementation and maps `SCALP_PING_5S_D_*`
env keys to `SCALP_PING_5S_C_*` before loading the runtime module.
"""

from __future__ import annotations

import asyncio
import os

from workers.scalp_ping_5s_d.worker import _apply_alt_env


def _apply_exit_env() -> None:
    _apply_alt_env(
        "SCALP_PING_5S_D",
        fallback_tag="scalp_ping_5s_d_live",
        fallback_log_prefix="[SCALP_PING_5S_D]",
    )

    # Ensure C exit worker consumes D strategy tags in this process.
    d_exit_tags = os.getenv("SCALP_PING_5S_D_EXIT_TAGS", "").strip()
    if d_exit_tags:
        os.environ["SCALP_PING_5S_C_EXIT_TAGS"] = d_exit_tags
        os.environ["SCALP_PRECISION_EXIT_TAGS"] = d_exit_tags

    d_pocket = os.getenv("SCALP_PING_5S_D_POCKET", "").strip()
    if d_pocket:
        os.environ["SCALP_PING_5S_C_POCKET"] = d_pocket
        os.environ["SCALP_PRECISION_POCKET"] = d_pocket


_apply_exit_env()

from workers.scalp_ping_5s_c.exit_worker import RangeFaderExitWorker  # noqa: E402


async def scalp_ping_5s_d_exit_worker() -> None:
    worker = RangeFaderExitWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(scalp_ping_5s_d_exit_worker())
