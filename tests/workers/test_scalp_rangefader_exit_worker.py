from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from workers.scalp_rangefader import exit_worker as range_exit_worker


class _DummyPositionManager:
    def close(self) -> None:
        return None


def _make_worker(monkeypatch):
    monkeypatch.setattr(range_exit_worker, "PositionManager", _DummyPositionManager)
    worker = range_exit_worker.RangeFaderExitWorker()
    worker.min_hold_sec = 1.0
    worker.loss_cut_hard_pips = 2.0
    worker.loss_cut_max_hold_sec = 60.0
    worker.close_retry_cooldown_sec = 60.0

    async def _fake_pro_stop(*_args, **_kwargs):
        return False

    monkeypatch.setattr(range_exit_worker, "maybe_close_pro_stop", _fake_pro_stop)
    monkeypatch.setattr(
        range_exit_worker,
        "decide_reentry",
        lambda **_kwargs: SimpleNamespace(action="skip", shadow=False),
    )
    monkeypatch.setattr(range_exit_worker, "all_factors", lambda: {"M1": {}, "H4": {}})
    monkeypatch.setattr(
        range_exit_worker,
        "apply_exit_forecast_to_loss_cut",
        lambda *, soft_pips, hard_pips, max_hold_sec, adjustment, floor_pips: (
            soft_pips,
            hard_pips,
            max_hold_sec,
        ),
    )
    monkeypatch.setattr(
        range_exit_worker,
        "apply_exit_forecast_to_targets",
        lambda **kwargs: (
            kwargs["profit_take"],
            kwargs["trail_start"],
            kwargs["trail_backoff"],
            kwargs["lock_buffer"],
        ),
    )
    return worker


def test_negative_close_failure_uses_retry_cooldown(monkeypatch):
    worker = _make_worker(monkeypatch)
    now_monotonic = {"value": 100.0}
    monkeypatch.setattr(range_exit_worker.time, "monotonic", lambda: now_monotonic["value"])
    monkeypatch.setattr(range_exit_worker, "mark_pnl_pips", lambda *_args, **_kwargs: -3.2)

    attempts: list[str] = []

    async def _fake_close(trade_id, units, reason, pnl, client_order_id, allow_negative=False, **_kwargs):
        attempts.append(str(reason))
        return False

    monkeypatch.setattr(worker, "_close", _fake_close)

    now = datetime.now(timezone.utc)
    trade = {
        "trade_id": "T-retry-blocked",
        "units": 1000,
        "price": 150.100,
        "open_time": (now - timedelta(minutes=5)).isoformat(),
        "client_order_id": "cid-retry-blocked",
        "entry_thesis": {"strategy_tag": "RangeFader-buy-fade"},
    }

    asyncio.run(worker._review_trade(trade, now, mid=150.030, range_active=False))
    now_monotonic["value"] = 120.0
    asyncio.run(worker._review_trade(trade, now, mid=150.030, range_active=False))
    now_monotonic["value"] = 161.0
    asyncio.run(worker._review_trade(trade, now, mid=150.030, range_active=False))

    assert attempts == ["max_adverse", "max_adverse"]
    state = worker._states.get("T-retry-blocked")
    assert state is not None
    assert state.close_retry_reason == "max_adverse"


def test_negative_close_success_clears_worker_state(monkeypatch):
    worker = _make_worker(monkeypatch)
    monkeypatch.setattr(range_exit_worker, "mark_pnl_pips", lambda *_args, **_kwargs: -3.2)

    attempts: list[str] = []

    async def _fake_close(trade_id, units, reason, pnl, client_order_id, allow_negative=False, **_kwargs):
        attempts.append(str(reason))
        return True

    monkeypatch.setattr(worker, "_close", _fake_close)

    now = datetime.now(timezone.utc)
    trade = {
        "trade_id": "T-close-ok",
        "units": 1000,
        "price": 150.100,
        "open_time": (now - timedelta(minutes=5)).isoformat(),
        "client_order_id": "cid-close-ok",
        "entry_thesis": {"strategy_tag": "RangeFader-sell-fade"},
    }

    asyncio.run(worker._review_trade(trade, now, mid=150.030, range_active=False))

    assert attempts == ["max_adverse"]
    assert "T-close-ok" not in worker._states
