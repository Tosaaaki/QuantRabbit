from __future__ import annotations

import os

import pytest

os.environ.setdefault("DISABLE_GCP_SECRET_MANAGER", "1")


def _tick(epoch: float, bid: float, ask: float, mid: float) -> dict[str, float]:
    return {"epoch": epoch, "bid": bid, "ask": ask, "mid": mid}


def test_build_tick_signal_detects_long(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.time, "time", lambda: 101.0)
    rows = [
        _tick(100.0, 150.000, 150.002, 150.001),
        _tick(100.2, 150.002, 150.004, 150.003),
        _tick(100.4, 150.004, 150.006, 150.005),
        _tick(100.6, 150.006, 150.008, 150.007),
        _tick(100.8, 150.008, 150.010, 150.009),
        _tick(101.0, 150.010, 150.012, 150.011),
    ]

    monkeypatch.setattr(worker.config, "MIN_TICKS", 6)
    monkeypatch.setattr(worker.config, "MIN_SIGNAL_TICKS", 4)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_SEC", 1.2)
    monkeypatch.setattr(worker.config, "MAX_TICK_AGE_MS", 900.0)
    monkeypatch.setattr(worker.config, "MOMENTUM_TRIGGER_PIPS", 0.7)
    monkeypatch.setattr(worker.config, "MOMENTUM_SPREAD_MULT", 0.6)
    monkeypatch.setattr(worker.config, "IMBALANCE_MIN", 0.58)
    monkeypatch.setattr(worker.config, "MIN_TICK_RATE", 3.0)

    sig = worker._build_tick_signal(rows, spread_pips=0.2)
    assert sig is not None
    assert sig.side == "long"
    assert sig.momentum_pips > 0
    assert sig.confidence >= worker.config.CONFIDENCE_FLOOR


def test_build_tick_signal_rejects_low_imbalance(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.time, "time", lambda: 201.0)
    rows = [
        _tick(200.0, 150.000, 150.002, 150.001),
        _tick(200.2, 150.001, 150.003, 150.002),
        _tick(200.4, 150.000, 150.002, 150.001),
        _tick(200.6, 150.001, 150.003, 150.002),
        _tick(200.8, 150.000, 150.002, 150.001),
        _tick(201.0, 150.001, 150.003, 150.002),
    ]

    monkeypatch.setattr(worker.config, "MIN_TICKS", 6)
    monkeypatch.setattr(worker.config, "MIN_SIGNAL_TICKS", 4)
    monkeypatch.setattr(worker.config, "SIGNAL_WINDOW_SEC", 1.2)
    monkeypatch.setattr(worker.config, "MAX_TICK_AGE_MS", 900.0)
    monkeypatch.setattr(worker.config, "MOMENTUM_TRIGGER_PIPS", 0.4)
    monkeypatch.setattr(worker.config, "MOMENTUM_SPREAD_MULT", 0.4)
    monkeypatch.setattr(worker.config, "IMBALANCE_MIN", 0.85)
    monkeypatch.setattr(worker.config, "MIN_TICK_RATE", 2.0)

    sig = worker._build_tick_signal(rows, spread_pips=0.2)
    assert sig is None


def test_compute_targets_is_spread_aware() -> None:
    from workers.scalp_ping_5s import worker

    tp_pips, sl_pips = worker._compute_targets(spread_pips=0.9, momentum_pips=2.4)
    assert tp_pips >= min(
        worker.config.TP_MAX_PIPS,
        max(worker.config.TP_BASE_PIPS, 0.9 + worker.config.TP_NET_MIN_PIPS),
    )
    assert tp_pips <= worker.config.TP_MAX_PIPS
    assert sl_pips >= (0.9 * worker.config.SL_SPREAD_MULT + worker.config.SL_SPREAD_BUFFER_PIPS)
    assert sl_pips <= worker.config.SL_MAX_PIPS


def test_compute_targets_uses_spread_plus_micro_edge(monkeypatch) -> None:
    from workers.scalp_ping_5s import worker

    monkeypatch.setattr(worker.config, "TP_BASE_PIPS", 0.2)
    monkeypatch.setattr(worker.config, "TP_NET_MIN_PIPS", 0.25)
    monkeypatch.setattr(worker.config, "TP_MOMENTUM_BONUS_MAX", 0.0)
    monkeypatch.setattr(worker.config, "TP_MAX_PIPS", 1.0)

    tp_pips, _ = worker._compute_targets(spread_pips=0.30, momentum_pips=0.1)
    assert tp_pips == pytest.approx(0.55, abs=1e-6)
