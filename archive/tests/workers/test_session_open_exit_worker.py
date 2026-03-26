from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest


class _DummyPosManager:
    def close(self) -> None:
        return None


def _sample_trade(
    *, opened_sec: float, pnl_pips: float, strategy_tag: str = "session_open_breakout"
) -> dict:
    return {
        "trade_id": "session-open-trade-1",
        "units": 1000,
        "price": 158.000,
        "open_time": (
            datetime.now(timezone.utc) - timedelta(seconds=opened_sec)
        ).isoformat(),
        "client_order_id": "qr-test-session-open",
        "entry_thesis": {
            "strategy_tag": strategy_tag,
        },
        "stop_loss": {"price": 157.955},
        "take_profit": {"price": 158.050},
        "unrealized_pl_pips": pnl_pips,
    }


def _build_worker(monkeypatch: pytest.MonkeyPatch):
    import workers.session_open.exit_worker as exit_worker

    monkeypatch.setattr(exit_worker, "PositionManager", lambda: _DummyPosManager())
    monkeypatch.setattr(
        exit_worker, "build_exit_forecast_adjustment", lambda **_kwargs: None
    )
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
    monkeypatch.setattr(
        exit_worker,
        "detect_range_mode",
        lambda *_args, **_kwargs: type("R", (), {"active": False})(),
    )
    monkeypatch.setattr(exit_worker, "all_factors", lambda: {})
    worker = exit_worker.SessionOpenExitWorker()
    worker.min_hold_sec = 300.0
    worker.loss_cut_min_hold_sec = 600.0
    return exit_worker, worker


def test_session_open_moves_broker_protection_before_min_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(
        exit_worker,
        "be_profile_for_tag",
        lambda _tag: {
            "trigger_pips": 1.4,
            "lock_ratio": 0.38,
            "min_lock_pips": 0.45,
            "cooldown_sec": 0.0,
        },
    )
    monkeypatch.setattr(
        exit_worker,
        "tp_move_profile_for_tag",
        lambda _tag: {
            "enabled": True,
            "trigger_pips": 1.4,
            "buffer_pips": 0.8,
            "min_gap_pips": 0.3,
        },
    )
    set_protections = AsyncMock(return_value=True)
    monkeypatch.setattr(exit_worker, "set_trade_protections", set_protections)
    close_mock = AsyncMock()
    monkeypatch.setattr(worker, "_close", close_mock)

    asyncio.run(
        worker._review_trade(
            _sample_trade(opened_sec=90, pnl_pips=3.0),
            now=datetime.now(timezone.utc),
            mid=158.030,
            range_active=False,
        )
    )

    set_protections.assert_awaited_once_with(
        "session-open-trade-1",
        sl_price=158.027,
        tp_price=158.038,
    )
    close_mock.assert_not_called()
