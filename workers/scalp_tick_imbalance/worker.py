"""Scalp Tick Imbalance entry worker wrapper."""

from __future__ import annotations

import asyncio
import logging
import os
import importlib


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

    from workers.scalp_precision import worker as precision_worker

    importlib.reload(precision_worker.config)
    importlib.reload(precision_worker)
    asyncio.run(precision_worker.scalp_precision_worker())


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


if __name__ == "__main__":
    _configure_logging()
    _run_tick_imbalance_strategy()
