from __future__ import annotations

import asyncio


async def fetch_loop(*args, **kwargs):  # pragma: no cover - offline stub
    while True:
        await asyncio.sleep(300)

