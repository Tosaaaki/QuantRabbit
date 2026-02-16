"""Scalp Ping 5s dedicated EXIT worker.

Maps SCALP_PING_5S_* settings to precision-exit settings and runs the
scalp_precision exit flow in-process after prefix mapping.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path


def _apply_prefix_map(prefix: str) -> None:
    prefix = prefix.rstrip("_")
    source = f"{prefix}_"
    target = "SCALP_PRECISION_"
    for key, value in list(os.environ.items()):
        key = str(key)
        if not key.startswith(source):
            continue
        os.environ[f"{target}{key[len(source):]}"] = str(value)

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


def _run_exit_worker() -> None:
    _ENV_PREFIX = "SCALP_PING_5S"
    _apply_prefix_map(_ENV_PREFIX)

    if __package__ in (None, ""):
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    from workers.scalp_precision import exit_worker

    asyncio.run(exit_worker.scalp_precision_exit_worker())


if __name__ == "__main__":
    _configure_logging()
    _run_exit_worker()
