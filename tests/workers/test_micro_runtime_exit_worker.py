from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest


class _DummyPosManager:
    def close(self) -> None:
        return None


def _sample_trade() -> dict:
    return {
        "trade_id": "454064",
        "units": -5422,
        "price": 158.394,
        "open_time": (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat(),
        "entry_thesis": {
            "strategy_tag": "MomentumBurst",
            "tp_pips": 8.374,
        },
        "client_order_id": "qr-test-momentum",
    }


def _build_worker(monkeypatch: pytest.MonkeyPatch):
    import workers.micro_runtime.exit_worker as exit_worker

    monkeypatch.setattr(exit_worker, "PositionManager", lambda: _DummyPosManager())
    worker = exit_worker.MicroMultiExitWorker()
    monkeypatch.setattr(exit_worker, "momentum_scale", lambda **_kwargs: (1.0, {}))
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
    monkeypatch.setattr(exit_worker, "maybe_close_pro_stop", AsyncMock(return_value=False))
    monkeypatch.setattr(exit_worker, "_exit_candle_reversal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exit_worker, "_reversion_kind", lambda *_args, **_kwargs: None)
    return exit_worker, worker


def test_rsi_take_waits_for_strategy_min_profit(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(exit_worker, "mark_pnl_pips", lambda *_args, **_kwargs: 1.5)
    monkeypatch.setattr(exit_worker, "exit_profile_for_tag", lambda _tag: {"rsi_take_min_pips": 1.6})

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
            _sample_trade(),
            now=datetime.now(timezone.utc),
            mid=158.380,
            rsi=23.0,
            range_active=False,
            adx=39.0,
            bbw=0.0026,
        )
    )

    assert calls == []


def test_rsi_take_closes_once_profit_buffer_is_met(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(exit_worker, "mark_pnl_pips", lambda *_args, **_kwargs: 1.8)
    monkeypatch.setattr(exit_worker, "exit_profile_for_tag", lambda _tag: {"rsi_take_min_pips": 1.6})

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
            _sample_trade(),
            now=datetime.now(timezone.utc),
            mid=158.377,
            rsi=23.0,
            range_active=False,
            adx=39.0,
            bbw=0.0026,
        )
    )

    assert calls == [("rsi_take", 1.8)]
