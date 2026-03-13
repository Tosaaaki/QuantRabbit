from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest


class _DummyPosManager:
    def close(self) -> None:
        return None


def _sample_trade(*, opened_sec: float, pnl_pips: float, strategy_tag: str = "WickReversalBlend") -> dict:
    return {
        "trade_id": "wick-trade-1",
        "units": 1000,
        "price": 158.000,
        "open_time": (datetime.now(timezone.utc) - timedelta(seconds=opened_sec)).isoformat(),
        "client_order_id": "qr-test-wick",
        "entry_thesis": {
            "strategy_tag": strategy_tag,
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


def test_precision_exit_worker_moves_broker_sl_tp_once_trade_is_in_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(
        exit_worker,
        "_exit_profile_for_tag",
        lambda _tag: {
            "min_hold_sec": 0.0,
            "profit_pips": 2.4,
            "trail_start_pips": 1.8,
            "trail_backoff_pips": 0.5,
            "lock_buffer_pips": 0.2,
        },
    )
    monkeypatch.setattr(
        exit_worker,
        "_be_profile_for_tag",
        lambda _tag, *, pocket: {
            "trigger_pips": 1.0,
            "lock_ratio": 0.42,
            "min_lock_pips": 0.35,
            "cooldown_sec": 10.0,
        },
    )
    monkeypatch.setattr(
        exit_worker,
        "_tp_move_profile_for_tag",
        lambda _tag, *, pocket: {
            "enabled": True,
            "trigger_pips": 0.85,
            "buffer_pips": 0.55,
            "min_gap_pips": 0.3,
        },
    )
    set_protections = AsyncMock(return_value=True)
    monkeypatch.setattr(exit_worker, "set_trade_protections", set_protections)

    trade = _sample_trade(opened_sec=120, pnl_pips=1.4, strategy_tag="PrecisionLowVol")
    trade["stop_loss"] = {"price": 157.980}
    trade["take_profit"] = {"price": 158.030}

    asyncio.run(
        worker._review_trade(
            trade,
            now=datetime.now(timezone.utc),
            mid=158.014,
            range_active=False,
        )
    )

    set_protections.assert_awaited_once_with(
        "wick-trade-1",
        sl_price=158.006,
        tp_price=158.02,
    )


def test_precision_exit_worker_tightens_protection_under_headwind_and_wide_spread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import workers.scalp_wick_reversal_blend.exit_worker as exit_worker

    neutral = exit_worker._wick_live_protection_adjustments(
        thesis={"atr_pips": 1.2},
        fac_m1={"atr_pips": 1.2},
        range_active=False,
        bid=158.008,
        ask=158.019,
    )
    stressed = exit_worker._wick_live_protection_adjustments(
        thesis={
            "atr_pips": 1.2,
            "continuation_pressure": 0.92,
            "setup_quality": 0.40,
            "reversion_support": 0.18,
            "range_reason": "volatility_compression",
            "flow_headwind_regime": "continuation_headwind",
            "projection": {"score": -1.0},
        },
        fac_m1={"atr_pips": 1.2},
        range_active=False,
        bid=158.008,
        ask=158.019,
    )

    assert stressed["trigger_mult"] < neutral["trigger_mult"]
    assert stressed["lock_ratio_mult"] > neutral["lock_ratio_mult"]
    assert stressed["buffer_mult"] < neutral["buffer_mult"]


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


def test_wick_reversal_blend_dynamic_exit_respects_entry_quality_before_taking_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(
        exit_worker,
        "_exit_profile_for_tag",
        lambda _tag: {
            "min_hold_sec": 0.0,
            "profit_pips": 1.2,
            "trail_start_pips": 1.7,
            "trail_backoff_pips": 0.55,
            "lock_buffer_pips": 0.25,
            "loss_cut_hard_pips": 6.0,
            "loss_cut_max_hold_sec": 420.0,
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

    trade = _sample_trade(opened_sec=120, pnl_pips=1.9)
    trade["entry_thesis"].update(
        {
            "sl_pips": 2.0,
            "tp_pips": 3.0,
            "atr_pips": 2.2,
            "wick_blend_quality": 0.62,
        }
    )

    now = datetime.now(timezone.utc)
    asyncio.run(worker._review_trade(trade, now=now, mid=158.019, range_active=False))

    assert calls == []

    trade["unrealized_pl_pips"] = 2.4
    asyncio.run(worker._review_trade(trade, now=now, mid=158.024, range_active=False))

    assert calls == [("take_profit", 2.4)]


def test_vwap_revert_dynamic_exit_uses_same_headwind_adjustment_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(
        exit_worker,
        "_exit_profile_for_tag",
        lambda _tag: {
            "min_hold_sec": 0.0,
            "profit_pips": 1.2,
            "trail_start_pips": 1.7,
            "trail_backoff_pips": 0.55,
            "lock_buffer_pips": 0.25,
            "loss_cut_hard_pips": 6.0,
            "loss_cut_max_hold_sec": 420.0,
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
    adjust_calls: list[str] = []

    def _fake_adjust(**kwargs):
        adjust_calls.append(str(kwargs["thesis"].get("strategy_tag")))
        return {
            "profit_take": 2.5,
            "trail_start": 1.8,
            "trail_backoff": 0.3,
            "lock_buffer": 0.2,
            "loss_cut_hard_pips": 2.0,
            "loss_cut_max_hold_sec": 180.0,
        }

    monkeypatch.setattr(exit_worker, "wick_blend_exit_adjustments", _fake_adjust)

    trade = _sample_trade(opened_sec=120, pnl_pips=1.9, strategy_tag="VwapRevertS")
    trade["entry_thesis"].update(
        {
            "sl_pips": 1.8,
            "tp_pips": 2.2,
            "atr_pips": 2.53,
            "continuation_pressure": 0.63,
            "rsi": 63.9,
            "adx": 21.4,
            "range_score": 0.37,
            "vwap_gap": 18.4,
            "projection": {"score": -0.12},
        }
    )

    now = datetime.now(timezone.utc)
    asyncio.run(worker._review_trade(trade, now=now, mid=158.019, range_active=False))

    assert calls == []
    assert adjust_calls == ["VwapRevertS", "VwapRevertS"]

    trade["unrealized_pl_pips"] = 2.6
    asyncio.run(worker._review_trade(trade, now=now, mid=158.022, range_active=False))

    assert calls == [("take_profit", 2.6)]
