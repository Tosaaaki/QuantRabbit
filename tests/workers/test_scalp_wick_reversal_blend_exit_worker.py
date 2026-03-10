from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest


class _DummyPosManager:
    def close(self) -> None:
        return None


def _sample_trade(*, opened_sec: float, pnl_pips: float) -> dict:
    return {
        "trade_id": "wick-trade-1",
        "units": 1000,
        "price": 158.000,
        "open_time": (datetime.now(timezone.utc) - timedelta(seconds=opened_sec)).isoformat(),
        "client_order_id": "qr-test-wick",
        "entry_thesis": {
            "strategy_tag": "WickReversalBlend",
        },
        "unrealized_pl_pips": pnl_pips,
    }


def _build_worker(monkeypatch: pytest.MonkeyPatch):
    import workers.scalp_wick_reversal_blend.exit_worker as exit_worker

    monkeypatch.setattr(exit_worker, "PositionManager", lambda: _DummyPosManager())
    monkeypatch.setattr(exit_worker, "load_rollout_start_ts", lambda *_args, **_kwargs: 0.0)
    worker = exit_worker.RangeFaderExitWorker()
    monkeypatch.setattr(exit_worker, "trade_passes_rollout", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(exit_worker, "build_exit_forecast_adjustment", lambda **_kwargs: None)
    monkeypatch.setattr(
        exit_worker,
        "apply_exit_forecast_to_targets",
        lambda **kwargs: (
            kwargs["profit_take"],
            kwargs["trail_start"],
            kwargs["trail_backoff"],
            kwargs["lock_buffer"],
        ),
    )
    monkeypatch.setattr(
        exit_worker,
        "apply_exit_forecast_to_loss_cut",
        lambda **kwargs: (
            kwargs["soft_pips"],
            kwargs["hard_pips"],
            kwargs["max_hold_sec"],
        ),
    )
    monkeypatch.setattr(exit_worker, "maybe_close_pro_stop", AsyncMock(return_value=False))
    monkeypatch.setattr(exit_worker, "_exit_candle_reversal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exit_worker, "_latest_bid_ask", lambda: (None, None))
    monkeypatch.setattr(exit_worker, "all_factors", lambda: {})
    monkeypatch.setattr(exit_worker, "log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exit_worker, "set_trade_protections", AsyncMock(return_value=False))
    return exit_worker, worker


def test_wick_reversal_blend_take_profit_uses_strategy_exit_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(
        exit_worker,
        "_exit_profile_for_tag",
        lambda _tag: {
            "min_hold_sec": 0.0,
            "profit_pips": 0.8,
            "trail_start_pips": 4.0,
            "trail_backoff_pips": 0.4,
            "lock_buffer_pips": 0.2,
        },
    )

    calls: list[tuple[str, float]] = []

    async def _fake_close(
        _trade_id: str,
        _units: int,
        reason: str,
        pnl: float,
        _client_id: str,
        **_kwargs,
    ) -> None:
        calls.append((reason, pnl))

    monkeypatch.setattr(worker, "_close", _fake_close)

    asyncio.run(
        worker._review_trade(
            _sample_trade(opened_sec=120, pnl_pips=0.9),
            now=datetime.now(timezone.utc),
            mid=158.009,
            range_active=False,
        )
    )

    assert calls == [("take_profit", 0.9)]


def test_wick_reversal_blend_lock_floor_waits_for_profile_min_hold_before_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(
        exit_worker,
        "_exit_profile_for_tag",
        lambda _tag: {
            "min_hold_sec": 0.0,
            "profit_pips": 10.0,
            "trail_start_pips": 0.5,
            "trail_backoff_pips": 0.2,
            "lock_buffer_pips": 0.1,
            "lock_floor_min_hold_sec": 60.0,
        },
    )

    calls: list[tuple[str, float]] = []

    async def _fake_close(
        _trade_id: str,
        _units: int,
        reason: str,
        pnl: float,
        _client_id: str,
        **_kwargs,
    ) -> None:
        calls.append((reason, pnl))

    monkeypatch.setattr(worker, "_close", _fake_close)

    now = datetime.now(timezone.utc)
    asyncio.run(
        worker._review_trade(
            _sample_trade(opened_sec=20, pnl_pips=1.0),
            now=now,
            mid=158.010,
            range_active=False,
        )
    )
    asyncio.run(
        worker._review_trade(
            _sample_trade(opened_sec=20, pnl_pips=0.75),
            now=now,
            mid=158.008,
            range_active=False,
        )
    )

    assert calls == []

    asyncio.run(
        worker._review_trade(
            _sample_trade(opened_sec=90, pnl_pips=0.75),
            now=now,
            mid=158.008,
            range_active=False,
        )
    )

    assert calls == [("lock_floor", 0.75)]
