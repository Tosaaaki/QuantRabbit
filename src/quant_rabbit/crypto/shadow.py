from __future__ import annotations

import asyncio
import fcntl
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .fast import FastPaperConfig, FastPaperRunner, fast_report_markdown
from .ledger import CryptoLedger
from .outbox import AsyncTradeOutbox
from .paper import PaperEngine
from .report import atomic_write_json, atomic_write_text
from .stream import BitbankStreamError


@dataclass(frozen=True)
class PaperShadowServiceConfig:
    mode: str
    runtime_dir: Path
    initial_cash_jpy: Decimal = Decimal("10000")
    max_leverage: Decimal = Decimal("2")
    epoch_sec: float = 60.0
    max_events: int = 10_000_000
    progress_interval_sec: float = 5.0
    retry_delay_sec: float = 5.0


class PaperShadowAlreadyRunning(RuntimeError):
    pass


class PaperShadowService:
    """Long-running Public Stream Paper service with no broker dependency."""

    def __init__(
        self,
        config: PaperShadowServiceConfig,
        *,
        pairs: list[str],
        pair_fees: dict[str, tuple[Decimal, Decimal]],
        daily_interest_rates: dict[str, tuple[Decimal, Decimal]],
    ) -> None:
        if config.mode not in {"spot", "margin"}:
            raise ValueError("Paper Shadow mode must be spot or margin")
        self.config = config
        self.pairs = pairs
        self.pair_fees = pair_fees
        self.daily_interest_rates = daily_interest_rates
        self._stop_requested = False

    def run(self) -> int:
        runtime = self.config.runtime_dir
        runtime.mkdir(parents=True, exist_ok=True)
        lock_handle = (runtime / "service.lock").open("a+")
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            lock_handle.close()
            raise PaperShadowAlreadyRunning(
                f"{self.config.mode} Paper Shadow is already running"
            ) from exc
        started_at = datetime.now(timezone.utc).isoformat()
        ledger = CryptoLedger(runtime / "ledger.db")
        outbox = AsyncTradeOutbox(runtime / "trade_outbox.jsonl", ledger)
        paper = PaperEngine(
            ledger,
            initial_cash_jpy=self.config.initial_cash_jpy,
            allow_short=self.config.mode == "margin",
            max_leverage=(
                self.config.max_leverage
                if self.config.mode == "margin"
                else Decimal("1")
            ),
            trade_sink=outbox.enqueue,
        )
        previous_handlers = self._install_signal_handlers()

        def progress(payload: dict[str, Any]) -> None:
            state = {
                **payload,
                "service_pid": os.getpid(),
                "service_started_at_utc": started_at,
                "runtime_dir": str(runtime),
                "ledger_path": str(ledger.path),
                "outbox": outbox.status(),
                "stop_conditions": [
                    "launchd stop/unload",
                    "SIGTERM or SIGINT",
                    "unsafe environment fails before startup",
                    "exclusive service lock conflict",
                ],
            }
            atomic_write_json(runtime / "state.json", state)

        try:
            while not self._stop_requested:
                runner = FastPaperRunner(
                    ledger,
                    paper,
                    config=FastPaperConfig.from_env(),
                )
                try:
                    result = asyncio.run(
                        runner.run(
                            self.pairs,
                            self.pair_fees,
                            duration_sec=self.config.epoch_sec,
                            max_events=self.config.max_events,
                            daily_interest_rates=(
                                self.daily_interest_rates
                                if self.config.mode == "margin"
                                else {}
                            ),
                            progress_callback=progress,
                            progress_interval_sec=(
                                self.config.progress_interval_sec
                            ),
                        )
                    )
                    atomic_write_json(runtime / "latest_epoch.json", result)
                    atomic_write_text(
                        runtime / "latest_report.md",
                        fast_report_markdown(result),
                    )
                except (BitbankStreamError, OSError) as exc:
                    progress(
                        {
                            "schema": "QR_CRYPTO_PAPER_SHADOW_STATE_V1",
                            "status": "RETRYING_AFTER_PUBLIC_STREAM_ERROR",
                            "run_id": None,
                            "started_at_utc": started_at,
                            "heartbeat_at_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "mode": self.config.mode.upper(),
                            "pairs": self.pairs,
                            "events_processed": 0,
                            "actions": {},
                            "reasons": {type(exc).__name__: 1},
                            "fills": 0,
                            "books_ready": 0,
                            "guardian": {
                                "state": "RESTRICT",
                                "kill_switch": False,
                                "deterministic": True,
                            },
                            "metrics": paper.mark_to_market({}, {}),
                            "safety": runner.safety.as_dict(),
                            "progress_write_failures": 0,
                        }
                    )
                    self._wait_retry()
            return 0
        finally:
            outbox.close()
            self._restore_signal_handlers(previous_handlers)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    def _wait_retry(self) -> None:
        deadline = time.monotonic() + self.config.retry_delay_sec
        while not self._stop_requested and time.monotonic() < deadline:
            time.sleep(0.1)

    def _install_signal_handlers(self) -> dict[int, Any]:
        previous: dict[int, Any] = {}

        def request_stop(_signum: int, _frame: object) -> None:
            self._stop_requested = True

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        return previous

    @staticmethod
    def _restore_signal_handlers(previous: dict[int, Any]) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
