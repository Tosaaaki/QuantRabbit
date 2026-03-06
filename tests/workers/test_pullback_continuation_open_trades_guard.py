from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.scalp_pullback_continuation import worker as pullback_worker


class _DummyPositionManager:
    def __init__(self, positions: dict | None = None, *, err: Exception | None = None) -> None:
        self._positions = positions or {}
        self._err = err

    def get_open_positions(self, include_unknown: bool = False) -> dict:
        if self._err is not None:
            raise self._err
        return self._positions


def test_open_trades_guard_blocks_when_limit_reached(monkeypatch) -> None:
    monkeypatch.setattr(pullback_worker.config, "MAX_OPEN_TRADES", 1)
    monkeypatch.setattr(
        pullback_worker.config,
        "FAIL_CLOSED_ON_POSITIONS_ERROR",
        True,
    )

    allowed, reason, count = pullback_worker._passes_open_trades_guard(
        _DummyPositionManager(
            {
                "scalp": {
                    "open_trades": [
                        {"strategy_tag": "PullbackContinuation"},
                    ]
                }
            }
        ),
        "PullbackContinuation",
    )

    assert allowed is False
    assert reason == "max_open_trades"
    assert count == 1


def test_open_trades_guard_fails_closed_on_positions_error(monkeypatch) -> None:
    monkeypatch.setattr(pullback_worker.config, "MAX_OPEN_TRADES", 1)
    monkeypatch.setattr(
        pullback_worker.config,
        "FAIL_CLOSED_ON_POSITIONS_ERROR",
        True,
    )

    allowed, reason, count = pullback_worker._passes_open_trades_guard(
        _DummyPositionManager(err=RuntimeError("position manager down")),
        "PullbackContinuation",
    )

    assert allowed is False
    assert reason.startswith("open_positions_error:")
    assert count == -1
