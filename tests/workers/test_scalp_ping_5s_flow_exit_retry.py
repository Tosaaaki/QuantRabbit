from __future__ import annotations

import asyncio

from workers.scalp_ping_5s_flow import exit_worker as flow_exit_worker


class _DummyPositionManager:
    def close(self) -> None:
        return None


def _make_worker(monkeypatch):
    monkeypatch.setattr(flow_exit_worker, "PositionManager", _DummyPositionManager)
    worker = flow_exit_worker.RangeFaderExitWorker()
    worker.close_retry_cooldown_sec = 60.0
    return worker


def test_failed_close_uses_reason_scoped_retry_cooldown(monkeypatch):
    worker = _make_worker(monkeypatch)
    now_monotonic = {"value": 100.0}
    monkeypatch.setattr(
        flow_exit_worker.time, "monotonic", lambda: now_monotonic["value"]
    )

    attempts: list[str] = []

    async def _fake_close(
        trade_id, units, reason, pnl, client_order_id, allow_negative=False, **_kwargs
    ):
        attempts.append(str(reason))
        return False

    monkeypatch.setattr(worker, "_close", _fake_close)
    worker._direction_flip_states["T-flow-retry"] = (
        flow_exit_worker._DirectionFlipState(hits=2)
    )

    asyncio.run(
        worker._attempt_close(
            "T-flow-retry",
            -1000,
            "max_hold_loss",
            -0.4,
            "cid-flow-retry",
            allow_negative=True,
            clear_direction_flip=True,
        )
    )
    now_monotonic["value"] = 120.0
    asyncio.run(
        worker._attempt_close(
            "T-flow-retry",
            -1000,
            "max_hold_loss",
            -0.4,
            "cid-flow-retry",
            allow_negative=True,
            clear_direction_flip=True,
        )
    )
    now_monotonic["value"] = 161.0
    asyncio.run(
        worker._attempt_close(
            "T-flow-retry",
            -1000,
            "max_hold_loss",
            -0.4,
            "cid-flow-retry",
            allow_negative=True,
            clear_direction_flip=True,
        )
    )

    assert attempts == ["max_hold_loss", "max_hold_loss"]
    assert "T-flow-retry" in worker._states
    assert "T-flow-retry" in worker._direction_flip_states


def test_successful_close_clears_direction_flip_state(monkeypatch):
    worker = _make_worker(monkeypatch)

    attempts: list[str] = []

    async def _fake_close(
        trade_id, units, reason, pnl, client_order_id, allow_negative=False, **_kwargs
    ):
        attempts.append(str(reason))
        return True

    monkeypatch.setattr(worker, "_close", _fake_close)
    worker._direction_flip_states["T-flow-ok"] = flow_exit_worker._DirectionFlipState(
        hits=1
    )

    asyncio.run(
        worker._attempt_close(
            "T-flow-ok",
            -1000,
            "max_hold_loss",
            -0.4,
            "cid-flow-ok",
            allow_negative=True,
            clear_direction_flip=True,
        )
    )

    assert attempts == ["max_hold_loss"]
    assert "T-flow-ok" not in worker._states
    assert "T-flow-ok" not in worker._direction_flip_states
