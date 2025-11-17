from __future__ import annotations

from datetime import datetime, timedelta, timezone

from execution.exit_manager import ExitManager


def _base_open_info(side: str, opened: datetime, avg_price: float) -> dict:
    return {
        "open_trades": [
            {"side": side, "open_time": opened.isoformat()},
        ],
        f"{side}_avg_price": avg_price if side == "long" else None,
        f"{'short' if side == 'long' else 'long'}_avg_price": None,
        "avg_price": avg_price,
    }


def _fac_m1(close: float, ema: float) -> dict:
    return {
        "rsi": 70.0,
        "ema20": ema,
        "close": close,
        "atr_pips": 0.8,
        "candles": [{"close": close}] * 40,
    }


def test_micro_exit_guard_blocks_early_loss(monkeypatch):
    monkeypatch.setenv("EXIT_MICRO_MIN_HOLD_SEC", "80")
    monkeypatch.setenv("EXIT_MICRO_GUARD_LOSS_PIPS", "1.6")
    manager = ExitManager()
    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    opened = now - timedelta(seconds=20)
    open_info = _base_open_info("long", opened, avg_price=153.000)
    decision = manager._evaluate_long(
        pocket="micro",
        open_info=open_info,
        units=1000,
        reverse_signal=None,
        event_soon=False,
        rsi=70.0,
        ma10=153.01,
        ma20=153.02,
        adx=18.0,
        close_price=152.985,  # -1.5 pips
        ema20=152.99,
        range_mode=False,
        now=now,
        projection_primary=None,
        projection_fast=None,
        atr_pips=0.8,
        fac_m1=_fac_m1(152.985, 152.99),
        stage_tracker=None,
    )
    assert decision is None, "micro trades within guard window must not auto-close at small losses"


def test_micro_exit_guard_allows_exit_after_hold(monkeypatch):
    monkeypatch.setenv("EXIT_MICRO_MIN_HOLD_SEC", "40")
    monkeypatch.setenv("EXIT_MICRO_GUARD_LOSS_PIPS", "1.6")
    manager = ExitManager()
    now = datetime(2025, 1, 1, 12, 5, tzinfo=timezone.utc)
    opened = now - timedelta(seconds=120)
    open_info = _base_open_info("long", opened, avg_price=153.000)
    decision = manager._evaluate_long(
        pocket="micro",
        open_info=open_info,
        units=1000,
        reverse_signal=None,
        event_soon=False,
        rsi=70.0,
        ma10=153.01,
        ma20=153.02,
        adx=18.0,
        close_price=152.985,
        ema20=152.99,
        range_mode=False,
        now=now,
        projection_primary=None,
        projection_fast=None,
        atr_pips=0.8,
        fac_m1=_fac_m1(152.985, 152.99),
        stage_tracker=None,
    )
    assert decision is not None
    assert decision.reason == "rsi_overbought"


def test_scalp_exit_guard_blocks_early_loss(monkeypatch):
    monkeypatch.setenv("EXIT_SCALP_MIN_HOLD_SEC", "60")
    monkeypatch.setenv("EXIT_SCALP_GUARD_LOSS_PIPS", "2.2")
    manager = ExitManager()
    now = datetime(2025, 1, 1, 13, 0, tzinfo=timezone.utc)
    opened = now - timedelta(seconds=15)
    open_info = _base_open_info("short", opened, avg_price=153.20)
    decision = manager._evaluate_short(
        pocket="scalp",
        open_info=open_info,
        units=1000,
        reverse_signal=None,
        event_soon=False,
        rsi=35.0,
        ma10=153.18,
        ma20=153.12,
        adx=24.0,
        close_price=153.22,
        ema20=153.17,
        range_mode=False,
        now=now,
        projection_primary=None,
        projection_fast=None,
        atr_pips=3.0,
        fac_m1=_fac_m1(153.24, 153.17),
        stage_tracker=None,
    )
    assert decision is None
