"""Scalp Tick Imbalance dedicated ENTRY worker."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

_STRATEGY_MODE = "tick_imbalance"


def _configure_tick_imbalance_env() -> None:
    os.environ["SCALP_PRECISION_ENABLED"] = "1"
    os.environ["SCALP_PRECISION_MODE"] = _STRATEGY_MODE
    os.environ["SCALP_PRECISION_ALLOWLIST"] = _STRATEGY_MODE
    os.environ["SCALP_PRECISION_UNIT_ALLOWLIST"] = _STRATEGY_MODE
    os.environ["SCALP_PRECISION_MODE_FILTER_ALLOWLIST"] = "1"
    os.environ["SCALP_PRECISION_LOG_PREFIX"] = "[Scalp:TickImb]"


def _run_tick_imbalance_strategy() -> None:
    _configure_tick_imbalance_env()
    if __package__ in (None, ""):
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    from workers.scalp_precision import worker

    asyncio.run(worker.scalp_precision_worker())


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


if __name__ == "__main__":
    _configure_logging()
    _run_tick_imbalance_strategy()
