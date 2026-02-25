"""Wrapper EXIT worker for scalp_extrema_reversal.

Reuses the stable scalp_level_reject exit implementation by mapping env keys
before import-time constant resolution.
"""

from __future__ import annotations

import asyncio
import os


def _map_exit_env() -> None:
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_TAGS",
        os.getenv("SCALP_EXTREMA_REVERSAL_EXIT_TAGS", "scalp_extrema_reversal_live"),
    )
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_LOG_PREFIX",
        os.getenv(
            "SCALP_EXTREMA_REVERSAL_EXIT_LOG_PREFIX",
            "[ScalpExit:ExtremaReversal]",
        ),
    )
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_POCKET",
        os.getenv("SCALP_EXTREMA_REVERSAL_EXIT_POCKET", "scalp_fast"),
    )
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_PROFILE_ENABLED",
        os.getenv("SCALP_EXTREMA_REVERSAL_EXIT_PROFILE_ENABLED", "1"),
    )
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_PROFILE_PATH",
        os.getenv(
            "SCALP_EXTREMA_REVERSAL_EXIT_PROFILE_PATH",
            os.getenv("STRATEGY_PROTECTION_PATH", "config/strategy_exit_protections.yaml"),
        ),
    )
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_PROFILE_TTL_SEC",
        os.getenv("SCALP_EXTREMA_REVERSAL_EXIT_PROFILE_TTL_SEC", "12.0"),
    )


_map_exit_env()

from workers.scalp_level_reject.exit_worker import (  # noqa: E402
    scalp_level_reject_exit_worker,
)


async def scalp_extrema_reversal_exit_worker() -> None:
    await scalp_level_reject_exit_worker()


if __name__ == "__main__":
    asyncio.run(scalp_extrema_reversal_exit_worker())
