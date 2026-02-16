"""Scalp Wick Reversal Blend dedicated EXIT worker."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def _run_exit_worker() -> None:
    if __package__ in (None, ""):
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    from workers.scalp_precision import exit_worker

    asyncio.run(exit_worker.scalp_precision_exit_worker())


if __name__ == "__main__":
    _configure_logging()
    _run_exit_worker()
