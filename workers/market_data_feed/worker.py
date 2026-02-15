"""Dedicated market data feed worker.

Responsibility:
- Pull OANDA ticks via stream.
- Build live candles for M1/M5/H1/H4/D1.
- Update shared tick_window and factor_cache through the candle fetcher hooks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

from market_data.candle_fetcher import Candle, TimeFrame, initialize_history, start_candle_stream


LOG = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "on", "yes"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _active_timeframes() -> list[TimeFrame]:
    raw = os.getenv("MARKET_DATA_FEED_TIMEFRAMES", "M1,M5,H1,H4,D1")
    supported = ("M1", "M5", "H1", "H4", "D1")
    tfs: list[TimeFrame] = []
    for token in str(raw or "").replace("\n", ",").split(","):
        tf = token.strip().upper()
        if not tf:
            continue
        if tf not in supported:
            LOG.warning("[MARKET_DATA_FEED] unsupported timeframe: %s", tf)
            continue
        if tf not in tfs:
            tfs.append(tf)  # dedupe while keeping order
    if not tfs:
        tfs = list(supported)
    return tfs


async def _noop_handler(_: Candle) -> None:
    return None


def _build_handlers(timeframes: Sequence[TimeFrame]) -> List[Tuple[TimeFrame, Callable[[Candle], Any]]]:
    return [(tf, _noop_handler) for tf in timeframes]


async def market_data_feed_worker() -> None:
    if not _env_bool("MARKET_DATA_FEED_ENABLED", True):
        LOG.info("[MARKET_DATA_FEED] disabled by MARKET_DATA_FEED_ENABLED=0")
        while True:
            await asyncio.sleep(3600.0)

    instrument = os.getenv("MARKET_DATA_FEED_INSTRUMENT", "USD_JPY").strip().upper() or "USD_JPY"
    timeframes = _active_timeframes()
    handlers = _build_handlers(timeframes)
    retry_interval_sec = max(1.0, _env_float("MARKET_DATA_FEED_RETRY_SEC", 8.0))
    seed_interval_sec = max(60.0, _env_float("MARKET_DATA_FEED_SEED_INTERVAL_SEC", 900.0))
    last_seed_ts = 0.0

    loop_no = 0
    while True:
        loop_no += 1
        now_ts = time.monotonic()
        if now_ts - last_seed_ts >= seed_interval_sec:
            try:
                seeded = await initialize_history(instrument)
                if not seeded:
                    LOG.warning("[MARKET_DATA_FEED] seed incomplete for %s", instrument)
                last_seed_ts = time.monotonic()
            except Exception as exc:
                LOG.warning("[MARKET_DATA_FEED] initialize_history failed (%s): %s", instrument, exc)

        LOG.info(
            "[MARKET_DATA_FEED] loop=%d starting stream instrument=%s timeframes=%s",
            loop_no,
            instrument,
            ",".join(timeframes),
        )
        try:
            await start_candle_stream(instrument, handlers)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.error("[MARKET_DATA_FEED] stream failed (%s): %s", instrument, exc)
            await asyncio.sleep(retry_interval_sec)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


if __name__ == "__main__":
    _configure_logging()
    asyncio.run(market_data_feed_worker())
