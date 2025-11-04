"""
Minimal Kaizen audit loop stub.

This module acts as a safety net so that deployments missing the optional
Kaizen package do not cause the trading service to crash.  When the real
Kaizen implementation is available, it should replace this file.
"""

from __future__ import annotations

import asyncio
import logging

_DEFAULT_SLEEP_SECONDS = 900


async def audit_loop(sleep_seconds: int = _DEFAULT_SLEEP_SECONDS) -> None:
    """Background no-op worker that only emits a diagnostic once."""
    logging.info("[KAIZEN] stub audit loop running (no-op).")
    interval = max(60, int(sleep_seconds))
    while True:
        await asyncio.sleep(interval)
