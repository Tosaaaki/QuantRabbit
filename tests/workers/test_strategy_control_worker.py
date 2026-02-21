from __future__ import annotations

import pytest

from workers.strategy_control import worker as strategy_control_worker


def test_latest_tick_lag_ms_returns_none_without_ticks(monkeypatch) -> None:
    monkeypatch.setattr(
        strategy_control_worker.tick_window,
        "recent_ticks",
        lambda seconds=120.0, limit=1: [],
    )
    assert strategy_control_worker._latest_tick_lag_ms() is None


def test_latest_tick_lag_ms_calculates_from_tick_epoch(monkeypatch) -> None:
    monkeypatch.setattr(
        strategy_control_worker.tick_window,
        "recent_ticks",
        lambda seconds=120.0, limit=1: [{"epoch": 100.0}],
    )
    monkeypatch.setattr(strategy_control_worker.time, "time", lambda: 100.5)
    assert strategy_control_worker._latest_tick_lag_ms() == 500.0


def test_emit_slo_metrics_skips_when_market_closed(monkeypatch) -> None:
    calls: list[tuple[str, float, dict[str, str] | None]] = []

    monkeypatch.setattr(strategy_control_worker, "is_market_open", lambda: False)
    monkeypatch.setattr(
        strategy_control_worker,
        "log_metric",
        lambda metric, value, tags=None, ts=None: calls.append((metric, value, tags)),
    )

    emitted = strategy_control_worker._emit_slo_metrics(loop_start_mono=10.0)

    assert emitted is None
    assert calls == []


def test_emit_slo_metrics_logs_data_lag_and_latency(monkeypatch) -> None:
    calls: list[tuple[str, float, dict[str, str] | None]] = []

    monkeypatch.setattr(strategy_control_worker, "is_market_open", lambda: True)
    monkeypatch.setattr(strategy_control_worker, "_latest_tick_lag_ms", lambda: 321.0)
    monkeypatch.setattr(strategy_control_worker.time, "monotonic", lambda: 10.2)
    monkeypatch.setattr(
        strategy_control_worker,
        "log_metric",
        lambda metric, value, tags=None, ts=None: calls.append((metric, value, tags)),
    )

    emitted = strategy_control_worker._emit_slo_metrics(loop_start_mono=10.0)

    assert emitted is not None
    data_lag_ms, decision_latency_ms = emitted
    assert data_lag_ms == 321.0
    assert decision_latency_ms == pytest.approx(200.0)
    assert len(calls) == 2
    assert calls[0] == ("data_lag_ms", 321.0, {"mode": "strategy_control"})
    assert calls[1][0] == "decision_latency_ms"
    assert calls[1][1] == pytest.approx(200.0)
    assert calls[1][2] == {"mode": "strategy_control"}
