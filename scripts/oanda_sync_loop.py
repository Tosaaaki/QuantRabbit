#!/usr/bin/env python3
"""
Continuous OANDA → SQLite synchronizer.

Runs `PositionManager.sync_trades()` on a fixed interval so the local
`logs/trades.db` ledger stays aligned with the broker even when the main
trading loop is idle.

Example:
  python scripts/oanda_sync_loop.py --interval 45
  python scripts/oanda_sync_loop.py --once
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from contextlib import suppress

from execution.position_manager import PositionManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously sync trades from OANDA into logs/trades.db")
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between sync runs (default: 60).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sync cycle and exit.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Optional safety cap on loop iterations (0 = unlimited).",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    args = _parse_args()
    _configure_logging()

    # Keyboard interrupt handler
    stop_requested = False

    def _handle_signal(signum, frame):  # noqa: ANN001,ANN202
        nonlocal stop_requested
        logging.info("oanda_sync_loop received signal %s, shutting down...", signum)
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(Exception):
            signal.signal(sig, _handle_signal)

    pm = PositionManager()
    iterations = 0
    try:
        while not stop_requested:
            iterations += 1
            try:
                start = time.monotonic()
                pm.sync_trades()
                elapsed = time.monotonic() - start
                logging.info(
                    "[oanda_sync] iteration=%s duration=%.2fs last_tx_id=%s",
                    iterations,
                    elapsed,
                    pm._last_tx_id,  # noqa: SLF001 (debug output only)
                )
            except Exception as exc:  # noqa: BLE001
                logging.exception("[oanda_sync] sync failed: %s", exc)
            if args.once:
                break
            if args.max_iterations and iterations >= args.max_iterations:
                logging.info("[oanda_sync] reached iteration cap (%s); exiting.", args.max_iterations)
                break
            sleep_time = max(args.interval, 1.0)
            for _ in range(int(sleep_time)):
                if stop_requested:
                    break
                time.sleep(1)
            if stop_requested:
                break
            remaining = sleep_time - int(sleep_time)
            if remaining > 0 and not stop_requested:
                time.sleep(remaining)
    finally:
        with suppress(Exception):
            pm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
