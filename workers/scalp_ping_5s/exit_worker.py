"""Scalp Ping 5s dedicated EXIT worker.

Wraps `workers.scalp_precision.exit_worker` and maps SCALP_PING_5S_* settings
to the shared precision exit environment names.
"""

from __future__ import annotations

import asyncio
import logging
import os


def _apply_prefix_map(prefix: str) -> None:
    prefix = prefix.rstrip("_")
    source = f"{prefix}_"
    target = "SCALP_PRECISION_"
    for key, value in list(os.environ.items()):
        key = str(key)
        if not key.startswith(source):
            continue
        os.environ[f"{target}{key[len(source):]}"] = str(value)

    # Keep compatibility with common precision keys used by older env layouts.
    os.environ.setdefault(
        "SCALP_PRECISION_EXIT_TAGS",
        os.getenv(f"{prefix}_EXIT_TAGS", os.getenv(f"{prefix}_STRATEGY_TAG", "scalp_ping_5s_live")),
    )
    os.environ.setdefault("SCALP_PRECISION_POCKET", os.getenv(f"{prefix}_POCKET", "scalp_fast"))
    os.environ.setdefault(
        "SCALP_PRECISION_EXIT_LOG_PREFIX",
        os.getenv(f"{prefix}_EXIT_LOG_PREFIX", "[ScalpPing5S:Exit]"),
    )
    os.environ.setdefault("SCALP_PRECISION_EXIT_PROFILE_ENABLED", os.getenv(f"{prefix}_EXIT_PROFILE_ENABLED", "1"))


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


_ENV_PREFIX = "SCALP_PING_5S"
_apply_prefix_map(_ENV_PREFIX)

from workers.scalp_precision.exit_worker import scalp_precision_exit_worker


if __name__ == "__main__":
    _configure_logging()
    asyncio.run(scalp_precision_exit_worker())
