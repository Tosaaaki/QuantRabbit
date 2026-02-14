"""Scalp Wick Reversal Pro dedicated EXIT worker wrapper."""

from __future__ import annotations

import asyncio
import logging

from workers.scalp_precision.exit_worker import scalp_precision_exit_worker


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


if __name__ == "__main__":
    _configure_logging()
    asyncio.run(scalp_precision_exit_worker())
