from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class _DummyPosManager:
    def close(self) -> None:
        return None


def _sample_trade(
    *,
    opened_sec: float,
    pnl_pips: float,
    strategy_tag: str = "scalp_extrema_reversal_live",
) -> dict:
    return {
        "trade_id": "extrema-trade-1",
        "units": 1000,
        "price": 159.000,
        "open_time": (
            datetime.now(timezone.utc) - timedelta(seconds=opened_sec)
        ).isoformat(),
        "client_order_id": "qr-test-extrema",
        "entry_thesis": {
            "strategy_tag": strategy_tag,
        },
        "unrealized_pl_pips": pnl_pips,
    }


def _build_worker(monkeypatch: pytest.MonkeyPatch):
    import workers.scalp_level_reject.exit_worker as exit_worker

    monkeypatch.setattr(exit_worker, "PositionManager", lambda: _DummyPosManager())
    monkeypatch.setattr(
        exit_worker, "load_rollout_start_ts", lambda *_args, **_kwargs: 0.0
    )
    worker = exit_worker.RangeFaderExitWorker()
    monkeypatch.setattr(
        exit_worker, "trade_passes_rollout", lambda *_args, **_kwargs: True
    )
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
        exit_worker, "maybe_close_pro_stop", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        exit_worker, "_exit_candle_reversal", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(exit_worker, "_latest_bid_ask", lambda: (None, None))
    monkeypatch.setattr(exit_worker, "all_factors", lambda: {})
    monkeypatch.setattr(exit_worker, "log_metric", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        exit_worker, "set_trade_protections", AsyncMock(return_value=False)
    )
    return exit_worker, worker


def test_level_reject_exit_worker_moves_broker_sl_tp_for_extrema_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(
        exit_worker,
        "_exit_profile_for_tag",
        lambda _tag: {
            "min_hold_sec": 0.0,
            "profit_pips": 2.5,
            "trail_start_pips": 1.6,
            "trail_backoff_pips": 0.45,
            "lock_buffer_pips": 0.2,
        },
    )
    monkeypatch.setattr(
        exit_worker,
        "_be_profile_for_tag",
        lambda _tag, *, pocket: {
            "trigger_pips": 0.9,
            "lock_ratio": 0.45,
            "min_lock_pips": 0.35,
            "cooldown_sec": 8.0,
        },
    )
    monkeypatch.setattr(
        exit_worker,
        "_tp_move_profile_for_tag",
        lambda _tag, *, pocket: {
            "enabled": True,
            "trigger_pips": 0.8,
            "buffer_pips": 0.55,
            "min_gap_pips": 0.3,
        },
    )
    set_protections = AsyncMock(return_value=True)
    monkeypatch.setattr(exit_worker, "set_trade_protections", set_protections)

    trade = _sample_trade(opened_sec=90, pnl_pips=1.1)
    trade["stop_loss"] = {"price": 158.980}
    trade["take_profit"] = {"price": 159.030}

    asyncio.run(
        worker._review_trade(
            trade,
            now=datetime.now(timezone.utc),
            mid=159.012,
            range_active=False,
        )
    )

    set_protections.assert_awaited_once_with(
        "extrema-trade-1",
        sl_price=159.005,
        tp_price=159.018,
    )


def test_level_reject_exit_worker_lets_supportive_extrema_run_more_than_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import workers.scalp_level_reject.exit_worker as exit_worker

    neutral = exit_worker._level_reject_live_protection_adjustments(
        thesis={
            "atr_pips": 1.3,
            "range_reason": "range",
            "range_score": 0.52,
        },
        fac_m1={"atr_pips": 1.3},
        range_active=False,
        side="long",
        bid=159.010,
        ask=159.014,
    )
    supportive = exit_worker._level_reject_live_protection_adjustments(
        thesis={
            "atr_pips": 1.3,
            "range_reason": "range",
            "range_score": 0.84,
            "extrema": {
                "supportive_long": True,
                "tick_strength": 0.95,
                "long_bounce_pips": 1.6,
                "dist_low_pips": 0.4,
                "long_setup_pressure": {
                    "sl_rate": 0.10,
                    "fast_sl_rate": 0.04,
                    "net_jpy": 38.0,
                },
            },
        },
        fac_m1={"atr_pips": 1.3},
        range_active=False,
        side="long",
        bid=159.010,
        ask=159.014,
    )

    assert supportive["trigger_mult"] > neutral["trigger_mult"]
    assert supportive["lock_ratio_mult"] < neutral["lock_ratio_mult"]
    assert supportive["buffer_mult"] > neutral["buffer_mult"]


def test_extrema_inventory_stress_exit_closes_stale_loser_when_margin_usage_is_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_worker, worker = _build_worker(monkeypatch)
    monkeypatch.setattr(
        exit_worker,
        "_exit_profile_for_tag",
        lambda _tag: {
            "min_hold_sec": 0.0,
            "inventory_stress": {
                "enabled": True,
                "min_hold_sec": 28.0,
                "max_hold_sec": 90.0,
                "time_loss_pips": 0.5,
                "loss_pips": 1.5,
                "loss_sl_mult": 0.70,
                "loss_cap_pips": 2.1,
                "cooldown_sec": 0.0,
                "health_buffer": 0.0,
                "free_margin_ratio": 0.0,
                "margin_usage_ratio": 0.82,
                "unrealized_dd_ratio": 0.0,
            },
        },
    )
    monkeypatch.setattr(
        exit_worker,
        "get_account_snapshot",
        lambda **_kwargs: SimpleNamespace(
            nav=1000.0,
            margin_used=860.0,
            health_buffer=0.21,
            free_margin_ratio=0.24,
            unrealized_pl=-21.0,
        ),
    )

    calls: list[tuple[str, float, bool]] = []

    async def _fake_close(
        _trade_id: str,
        _units: int,
        reason: str,
        pnl: float,
        _client_id: str,
        **kwargs,
    ) -> None:
        calls.append((reason, pnl, bool(kwargs.get("allow_negative"))))

    monkeypatch.setattr(worker, "_close", _fake_close)

    trade = _sample_trade(opened_sec=80, pnl_pips=-1.8)
    trade["entry_thesis"]["sl_pips"] = 2.0

    asyncio.run(
        worker._review_trade(
            trade,
            now=datetime.now(timezone.utc),
            mid=158.982,
            range_active=False,
        )
    )

    assert calls == [("margin_usage_high", -1.8, True)]
